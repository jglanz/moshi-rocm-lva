
#include <assert.h>
#include <math.h>

#include <iostream>

#define MOSHI_BUILD
#include <moshi/moshi.h>

// for src/context.h
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#ifdef ENABLE_REPLAY
#include "replay.h"
#define CAPTURE(...)
#define CAPTURE_GROUP(...)
#else
#ifdef ENABLE_CAPTURE
#include "src/ggml_cap.h"
#else
#define CAPTURE(...)
#define CAPTURE_GROUP(...)
#endif
#endif
#define ONCE(code) {static bool once=false; if (!once) {{code;}; once=true;}}
#define ON_NTH(nth, code) {static int count=0; if (count++ == (nth)) {code;}}

#include "../src/config.h"
#include "../src/context.h"
#include "../src/wav.h"
#include "../src/loader.h"
#include "../src/torch.h"
#include "../src/moshi/modules/transformer.h"
#include "../src/moshi/utils/sampling.h"
#include "../src/moshi/models/lm_utils.h"
#include "../src/moshi/models/lm.h"
#include "../src/moshi/quantization/core_vq.h"
#include "../src/moshi/quantization/vq.h"
#include "../src/moshi/modules/conv.h"
#include "../src/moshi/modules/seanet.h"
#include "../src/moshi/models/compression.h"
#include "../src/moshi/models/lm_default.h"
#include "../src/moshi/models/tts.h"

struct moshi_context_t {
    ggml_backend * backend;
    ggml_backend * backend_cpu;
    own_ptr<ScratchContext> scratch_cpu;
    own_ptr<ScratchContext> scratch;
};

struct mimi_codec_t {
    int n_q;
    moshi_context_t * moshi;
    own_ptr<moshi_mimi_t> mimi;
    own_ptr<WeightLoader> mimi_weights;
};

struct mimi_encode_context_t {
    mimi_codec_t * codec;
    own_ptr<StateContext> state_ctx;
    own_ptr<moshi_mimi_state_t> states;
    std::vector<int> tokens;
    std::vector<float> frame;
};

struct mimi_decode_context_t {
    mimi_codec_t * codec;
    own_ptr<StateContext> state_ctx;
    own_ptr<moshi_mimi_state_t> states;
    std::vector<int> tokens;
    std::vector<float> frame;
};

struct tokenizer_t {
    sentencepiece::SentencePieceProcessor sp;
    int padding_between = 1;
    bool insert_bos = true;

    std::string tail;

    enum {
        FIND_START,
        FIND_END,
        CHECK_WORD,
        TOKENIZE,
    } state = FIND_START;
    int offset = 0, start_offset = 0, end_offset = 0;

    int found_break = 0;
    float time;
    std::deque<std::string> words;
};

// MARK: Moshi Context

moshi_context_t * moshi_alloc( ggml_backend * backend, ggml_backend * backend_cpu ) {
    assert( backend );
    assert( backend_cpu );
    auto dev_cpu = ggml_backend_get_device( backend_cpu );
    assert ( ggml_backend_dev_type( dev_cpu ) == GGML_BACKEND_DEVICE_TYPE_CPU );

    auto moshi = new moshi_context_t;
    moshi->backend_cpu = backend_cpu;
    moshi->backend = backend;
    moshi->scratch_cpu = new ScratchContext( 256, backend_cpu );
    moshi->scratch = new ScratchContext( 256, backend );
    return moshi;
}

void unref( moshi_context_t * moshi ) {
    delete moshi;
}

// MARK: Mimi Codec

static void mimi_alloc( mimi_codec_t * codec, moshi_context_t * moshi,
        const char * filename, int n_q ) {
    auto mimi = moshi_mimi_alloc_default( n_q );

    std::string filepath = filename;
    WeightLoader * mimi_weights;
    if ( filepath.ends_with( ".safetensors" ) ) {
        mimi_weights = WeightLoader::from_safetensor( filename,
            moshi->scratch_cpu, moshi->backend );
        if ( ! mimi_weights ) {
            fprintf(stderr, "error: mimi weights not found\n" );
            exit(1);
        }
    } else {
        mimi_weights = WeightLoader::from_gguf( filename,
            moshi->scratch_cpu, moshi->backend );
        if ( ! mimi_weights ) {
            fprintf(stderr, "error: mimi weights not found\n" );
            exit(1);
        }
        mimi_weights->load_gguf();
    }

    get_weights( mimi_weights, "mimi.quantizer.", mimi->quantizer );
    get_weights( mimi_weights, "mimi.upsample.convtr.", mimi->upsample );
    get_weights( mimi_weights, "mimi.decoder_transformer.transformer.", mimi->decoder_transformer );
    get_weights( mimi_weights, "mimi.decoder.", mimi->decoder );
    if ( mimi->encoder ) {
        get_weights( mimi_weights, "mimi.downsample.conv.", mimi->downsample );
        get_weights( mimi_weights, "mimi.encoder_transformer.transformer.", mimi->encoder_transformer );
        get_weights( mimi_weights, "mimi.encoder.", mimi->encoder );
    }
    if ( ! mimi_weights->is_gguf )
        mimi_weights->load();

    codec->n_q = n_q;
    codec->moshi = moshi;
    codec->mimi = mimi;
    codec->mimi_weights = mimi_weights;
}

mimi_codec_t * mimi_alloc( moshi_context_t * moshi, const char * filename, int n_q ) {
    auto codec = new mimi_codec_t;
    mimi_alloc( codec, moshi, filename, n_q );
    return codec;
}

void unref( mimi_codec_t * codec ) {
    delete codec;
}

float mimi_frame_rate( mimi_codec_t * codec ) {
    return codec->mimi->frame_rate;
}

int mimi_frame_size( mimi_codec_t * codec ) {
    return 1920;
}

void mimi_save_gguf( mimi_codec_t * codec, const char * filepath ) {
    codec->mimi_weights->save_gguf( filepath );
}

// MARK: Mimi Encode

static void mimi_encode_alloc_context( mimi_encode_context_t * context, mimi_codec_t * codec ) {
    auto state_ctx = new StateContext( codec->moshi->backend );
    auto mimi_states = moshi_mimi_encoder_states( state_ctx, codec->mimi );
    int frame_size = mimi_frame_size( codec );

    state_ctx->alloc();
    state_ctx->init();
    init( codec->moshi->scratch, mimi_states, codec->mimi );

    context->codec = codec;
    context->state_ctx = state_ctx;
    context->states = mimi_states;
    context->frame.resize( frame_size );
}

mimi_encode_context_t * mimi_encode_alloc_context( mimi_codec_t * codec ) {
    auto context = new mimi_encode_context_t;
    mimi_encode_alloc_context( context, codec );
    return context;
}

void unref( mimi_encode_context_t * context ) {
    delete context;
}

void mimi_encode_reset( mimi_encode_context_t * context ) {
    auto scratch = context->codec->moshi->scratch.ptr;
    auto mimi = context->codec->mimi.ptr;
    auto states = context->states.ptr;
    // state_ctx->init() FIRST, and it is not optional. init() below re-seeds the
    // two transformers' offsets and attention state, but the SEANet conv streaming
    // history lives as tensors in this context's StateContext and only
    // StateContext::init() puts those back. Without it a "reset" encoder carries
    // the previous stream's tail, and the same audio encodes to different codes --
    // which is exactly how this was found: a parity pass repeated after a reset
    // decoded a completely different token stream.
    context->state_ctx->init();
    init( scratch, states, mimi );
}

void mimi_encode_send( mimi_encode_context_t * context, float * frame ) {
    memcpy( context->frame.data(), frame, context->frame.size() * 4 );
}

void mimi_encode_receive( mimi_encode_context_t * context, int16_t * tokens ) {
    auto & ctx = *context->codec->moshi->scratch;
    auto mimi = context->codec->mimi.ptr;
    auto states = context->states.ptr;

    mimi_encode(
        ctx,
        mimi,
        states,
        context->frame,
        context->tokens
    );

    for ( int i = 0; i < context->tokens.size(); i++ )
        tokens[i] = context->tokens[i];
}

// MARK: Mimi Decode

static void mimi_decode_alloc_context( mimi_decode_context_t * context, mimi_codec_t * codec ) {
    auto state_ctx = new StateContext( codec->moshi->backend );
    NE upsample_ne = {1, 512, 1, 1};
    NE decoder_ne = {2, 512, 1, 1};
    auto mimi_states = moshi_mimi_states( state_ctx, codec->mimi, upsample_ne, decoder_ne );
    int frame_size = mimi_frame_size( codec );

    state_ctx->alloc();
    state_ctx->init();
    init( codec->moshi->scratch, mimi_states, codec->mimi );

    context->codec = codec;
    context->state_ctx = state_ctx;
    context->states = mimi_states;
    context->tokens.resize( codec->n_q );
    context->frame.resize( frame_size );
}

mimi_decode_context_t * mimi_decode_alloc_context( mimi_codec_t * codec ) {
    auto context = new mimi_decode_context_t;
    mimi_decode_alloc_context( context, codec );
    return context;
}

void unref( mimi_decode_context_t * context ) {
    delete context;
}

void mimi_decode_reset( mimi_decode_context_t * context ) {
    auto scratch = context->codec->moshi->scratch.ptr;
    auto mimi = context->codec->mimi.ptr;
    auto states = context->states.ptr;
    // See mimi_encode_reset: the conv streaming history needs StateContext::init(),
    // init() alone does not reach it.
    context->state_ctx->init();
    init( scratch, states, mimi );
}

void mimi_decode_send( mimi_decode_context_t * context, int16_t * tokens ) {
    auto n_q = context->codec->n_q;
    for ( int i = 0; i < n_q; i++ )
        context->tokens[i] = tokens[i];
}

void mimi_decode_receive( mimi_decode_context_t * context, float * frame ) {
    auto & ctx = *context->codec->moshi->scratch;
    auto mimi = context->codec->mimi.ptr;
    auto states = context->states.ptr;
    mimi_decode(
        ctx,
        mimi,
        states,
        context->tokens,
        context->frame
    );
    if ( frame )
        memcpy( frame, context->frame.data(), context->frame.size() * 4 );
}

// MARK: Voice Condition

static void voice_condition( voice_t * voice,
        conditioners_t * cond,
        ggml_tensor * speaker_wavs,
        moshi_context_t * moshi
) {
    ScratchContext & b_voice_ctx = *moshi->scratch.ptr;
    auto backend = moshi->backend;

    // cfg {'1.0': 0, '1.5': 1, '2.0': 2, '2.5': 3, '3.0': 4, '3.5': 5, '4.0': 6}
    auto cfg_val = b_voice_ctx.constant( 2 );
    auto cfg_emb = ggml_get_rows(b_voice_ctx, cond->cfg_embed_weight, cfg_val);
    auto cfg_cond = ggml_mul_mat(b_voice_ctx, cond->cfg_output_proj_weight, cfg_emb);

    // control {'ok': 0}
    auto control_val = b_voice_ctx.constant( 0 );
    auto control_emb = ggml_get_rows(b_voice_ctx, cond->control_embed_weight, control_val);
    auto control_cond = ggml_mul_mat(b_voice_ctx, cond->control_output_proj_weight, control_emb);

    auto condition_sum = ggml_add(b_voice_ctx, cfg_cond, control_cond);

    // speaker_wavs
    auto speaker_wavs_a = ggml_cont(b_voice_ctx, ggml_transpose(b_voice_ctx, speaker_wavs));
    auto speaker_wavs_b = ggml_mul_mat(b_voice_ctx, cond->speaker_wavs_output_proj_weight,
        speaker_wavs_a);
    //
    //auto speaker_wavs_cond = ggml_new_tensor_2d(b_voice_ctx, GGML_TYPE_F32,
    //    speaker_wavs_b->ne[0], speaker_wavs_b->ne[1] * 5);
    // fill with learnt padding
    //speaker_wavs_cond = ggml_scale_inplace(b_voice_ctx, speaker_wavs_cond, 0);
    //speaker_wavs_cond = ggml_add_inplace(b_voice_ctx, speaker_wavs_cond,
    //    speaker_wavs_learnt_padding);
    auto speaker_wavs_cond = ggml_repeat_4d( b_voice_ctx,
        cond->speaker_wavs_learnt_padding,
        speaker_wavs_b->ne[0], speaker_wavs_b->ne[1] * 5, 1, 1 );
    // set first speaker
    auto speaker_0 = ggml_view_2d(b_voice_ctx, speaker_wavs_cond,
        speaker_wavs_b->ne[0], speaker_wavs_b->ne[1],
        speaker_wavs_cond->nb[1], 0);
    speaker_0 = ggml_cpy(b_voice_ctx, speaker_wavs_b, speaker_0);
    speaker_wavs_cond = ggml_view_2d(b_voice_ctx, speaker_0,
        speaker_wavs_cond->ne[0], speaker_wavs_cond->ne[1],
        speaker_wavs_cond->nb[1], 0);

    float cross_attention_pos_emb_scale = 1;
    //auto positions = ggml_arange(b_voice_ctx, 0, speaker_wavs_cond->ne[1], 1);
    auto positions = b_voice_ctx.arange(0, (float) speaker_wavs_cond->ne[1], 1);
    auto pos_emb = ggml_timestep_embedding(b_voice_ctx, positions, 2048, 10000);
    auto condition_cross = ggml_add(b_voice_ctx, speaker_wavs_cond, ggml_scale(b_voice_ctx, pos_emb, cross_attention_pos_emb_scale));

    size_t mem_size = 2 * ggml_tensor_overhead();
    bool no_alloc = true;
    if (! backend ) {
        mem_size += ggml_nbytes( condition_sum );
        mem_size += ggml_nbytes( condition_cross );
        no_alloc = false;
    }
    voice->ctx = ggml_init({
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ no_alloc,
    });
    voice->sum = ggml_dup_tensor( voice->ctx, condition_sum );
    voice->cross = ggml_dup_tensor( voice->ctx, condition_cross );
    ggml_set_name( voice->sum, "sum" );
    ggml_set_name( voice->cross, "cross" );
    voice->buffer = ggml_backend_alloc_ctx_tensors( voice->ctx, backend );
    b_voice_ctx.build_forward_expand( condition_sum, voice->sum );
    b_voice_ctx.build_forward_expand( condition_cross, voice->cross );
    ONCE( b_voice_ctx.set_name("voice") );
    b_voice_ctx.compute();
}

// MARK: Tokenizer

tokenizer_t * tokenizer_alloc( const char * filepath, bool insert_bos ) {
    auto tok = new tokenizer_t;
    tok->sp.Load( filepath );
    tok->insert_bos = insert_bos;
    return tok;
}

void unref( tokenizer_t * tok ) {
    delete tok;
}

bool tokenizer_empty( tokenizer_t * tok ) {
    return ! tok->tail.size();
}

static int tokenizer_tokenize( tokenizer_t * tok, std::string & word, Entry * entry ) {
    const int text_bos_token = 1;

    std::vector<int> tokens;
    tok->sp.Encode(word, &tokens);
    if ( tok->insert_bos ) {
        tok->insert_bos = false;
        std::vector<int> new_tokens(1 + tokens.size());
        new_tokens[0] = text_bos_token;
        for (size_t i = 0; i < tokens.size(); i++)
            new_tokens[i+1] = tokens[i];
        tokens.swap( new_tokens );
    }
    int padding = 0;
    if ( tok->padding_between > 0 ) {
        padding = tok->padding_between + (int) tokens.size() - 1;
        if ( padding < 0 )
            padding = 0;
    }
    entry->tokens.swap( tokens );
    entry->text = word;
    entry->padding = padding;
    entry->time = ggml_time_ms();
    //queue_push( tok->outlet );
    //entries.push_back( Entry( tokens, word, padding ) );
    return 0;
}

static int tokenizer_break( tokenizer_t * tok, Entry * entry ) {
    const int tok_pad_id = 3;
    const float frame_rate = 12.5f;

    float time = tok->time;
    if ( time <= 0.f)
        return 0;

    if ( time > 10.f )
        time = 10.f;
    int npad = (int)( time * frame_rate );
    if ( npad == 0 )
        npad = 1;

    entry->text = "";
    while ( tok->words.size() ) {
        entry->text += tok->words.front() + " ";
        tok->words.pop_front();
    }
    entry->time = ggml_time_ms();
#ifdef OLD_WAY
    // tts.py: script_to_entries
    entry->tokens.reset();
    entry->padding = npad;
#else
    // tts_preprocess.rs: Tokenizer::  preprocess
    std::vector<int> tokens={ tok_pad_id, npad };
    entry->tokens.swap( tokens );
    entry->padding = 0;
#endif
    return 1;
}

static int tokenizer_break_time( tokenizer_t * tok ) {
    std::string & word = tok->words.back();
    if ( ! word.starts_with( "time=\"" ) )
        return 0;

    const_str_t s = { word.c_str(), (int)word.size() };
    int offset = 6;

    //int end_offset = str_find_not_of( s, offset, "0123456789." );
	char *end;
	const char *start = s.s + offset;
	tok->time = (float)strtod(start, &end);
	offset = (int) (end - start) + offset;

    int remaining = s.length - offset;
    if ( remaining < 2 || s.s[offset] != 's' || s.s[offset+1] != '"' )
        return 0;
    if ( remaining == 3 )
        return 0; // unexpected extra character

    if ( remaining >= 4 ) {
        if ( s.s[offset+2] != '/' || s.s[offset+3] != '>' )
            return 0; // expected "/>"
        return 3; // completed break
    }
    return 2; // incomplete break
}

int tokenizer_send( tokenizer_t * tok, std::string text ) {
    if ( text.size() == 0 ) {
        if ( tok->words.size() ) {
            tok->state = tokenizer_t::TOKENIZE;
        }
        // flush tail
        if ( tok->tail.size() )
            tok->tail += " ";
        return 0;
    }

    tok->tail += text;
    return 0;
}

int tokenizer_receive( tokenizer_t * tok, Entry * entry ) {
    const_str_t s = { tok->tail.c_str(), (int)tok->tail.size() };

    int found = 0;
    int & offset = tok->offset;
    int & start_offset = tok->start_offset;
    int & end_offset = tok->end_offset;

    while( true ) {
        switch( tok->state ) {
        case tokenizer_t::FIND_START:
            offset = str_skip_whitespaces( s, end_offset );
            if ( offset == s.length ) { // tail was all white space chars
                tok->tail = "";
                offset = 0;
                end_offset = 0;
                tok->state = tokenizer_t::FIND_START;
                return found;
            }
            start_offset = offset;
        case tokenizer_t::FIND_END:
            end_offset = str_find_whitespaces( s, offset );
            if ( end_offset == s.length ) { // maybe partial word
                offset = end_offset - start_offset;
                if ( start_offset != 0 ) {
                    tok->tail = tok->tail.substr( start_offset );
                    start_offset = 0;
                }
                tok->state = tokenizer_t::FIND_END;
                return found;
            }
            tok->words.push_back({});
            tok->words.back().assign( s.s + start_offset, end_offset - start_offset );
            if ( found ) {
                tok->state = tokenizer_t::CHECK_WORD;
                return found + 1;
            }
        case tokenizer_t::CHECK_WORD:
            if ( tok->found_break == 1 ) { // found break, check for time
                tok->found_break = tokenizer_break_time( tok );
                if ( tok->found_break == 0 ) { // no time property
                    tok->state = tokenizer_t::TOKENIZE;
                    break;
                }
                if ( tok->found_break == 2 ) { // need another word
                    tok->state = tokenizer_t::FIND_START;
                    break;
                }
                tok->state = tokenizer_t::CHECK_WORD;
                break;
            } else if ( tok->found_break == 2 ) {
                if ( tok->words.back().starts_with( "/>" ) ) {
                    tok->found_break = 3;
                    tok->state = tokenizer_t::CHECK_WORD;
                    break;
                } else {
                    tok->found_break = 0;
                    tok->state = tokenizer_t::TOKENIZE;
                    break;
                }
            } else if ( tok->found_break == 3 ) {
                int front_size = (int) tok->words.front().size();
                if ( front_size > 6 ) {
                    // must have token before break
                    auto word = tok->words.front().substr( 0, front_size - 6 );
                    tok->words.front() = tok->words.front().substr( front_size - 6 );
                    tok->words.push_front(word);
                    tok->state = tokenizer_t::TOKENIZE;
                    break;
                }
                int back_size = (int) tok->words.back().size();
                int tail = (int) tok->words.back().find( "/>" ) + 2;
                std::string word;
                if ( back_size != tail ) {
                    word = tok->words.back().substr( tail );
                    tok->words.back() = tok->words.back().substr( 0, tail );
                }
                tokenizer_break( tok, entry ); // consumes words
                tok->found_break = 0;
                if ( word.size() ) {
                    tok->words.push_back( word );
                    tok->state = tokenizer_t::CHECK_WORD;
                    return 2;
                }
                found = 1;
                tok->state = tokenizer_t::FIND_START;
                break;
            } else if ( tok->words.front().ends_with( "<break" ) ) {
                tok->found_break = 1;
                tok->state = tokenizer_t::FIND_START;
                break;
            }
        case tokenizer_t::TOKENIZE:
            tokenizer_tokenize( tok, tok->words.front(), entry );
            tok->words.pop_front();
            if ( tok->words.size() ) {
                tok->state = tokenizer_t::CHECK_WORD;
                return 2;
            }
            found = 1;
            tok->state = tokenizer_t::FIND_START;
        }
    }

    return 0;
}

std::string tokenizer_id_to_piece( tokenizer_t * tok, int token ) {
    return tok->sp.IdToPiece( token );
}

int tokenizer_encode( tokenizer_t * tok, const char * text, std::vector<int> & tokens ) {
    tokens.clear();
    if ( ! text || ! text[0] )
        return 0;
    tok->sp.Encode( std::string( text ), &tokens );
    return (int) tokens.size();
}

std::string tokenizer_id_to_text( tokenizer_t * tok, int token ) {
    // The SentencePiece piece with its word-initial marker U+2581 turned back into
    // a space.
    //
    // This matches the WHOLE three-byte sequence E2 96 81, where the copies open-
    // coded in tools/ (tools/moshi-stt.cpp:588 and three siblings,
    // tools/moshi-sts.cpp:917) test only the lead byte 0xE2 and then skip three
    // bytes unconditionally. 0xE2 leads every character in U+2000-U+2FFF, so those
    // copies silently replace an em dash, a curly quote, an ellipsis or a euro sign
    // with a space. Harmless for the ASCII-heavy en/fr vocabularies they were
    // written against; not something to propagate into a library function that a
    // transcript consumer will trust.
    const std::string piece = tok->sp.IdToPiece( token );
    std::string text;
    text.reserve( piece.size() );
    for ( size_t ci = 0; ci < piece.size(); ci++ ) {
        const unsigned char b0 = (unsigned char) piece[ci];
        if ( b0 == 0xE2 && ci + 2 < piece.size()
             && (unsigned char) piece[ci + 1] == 0x96
             && (unsigned char) piece[ci + 2] == 0x81 ) {
            text += ' ';
            ci += 2;
            continue;
        }
        text += piece[ci];
    }
    return text;
}

// MARK: LM

struct moshi_lm_t {
    std::string filepath;
    bool uses_cross;
    moshi_lmmodel_t * model;
    own_ptr<WeightLoader> weights;
    own_ptr<conditioners_t> cond;
    bool second_stream_ahead;
};

moshi_lm_t * moshi_lm_from_files(
    moshi_context_t * moshi,
    moshi_config_t * config,
    const char * filepath
) {
    auto lm = new moshi_lm_t;

    lm->filepath = filepath;

    if ( lm->filepath.ends_with( ".safetensors" ) ) {
        lm->weights = WeightLoader::from_safetensor( filepath, moshi->scratch_cpu, moshi->backend );
        if ( ! lm->weights )
            return NULL;
    } else {
        lm->weights = WeightLoader::from_gguf( filepath, moshi->scratch_cpu, moshi->backend );
        if ( ! lm->weights )
            return NULL;
    }

    lm->model = moshi_lmmodel_alloc_default( config );

    lm->uses_cross = config->cross_attention;
    lm->second_stream_ahead = config->tts_config.second_stream_ahead;

    return lm;
}

void unref( moshi_lm_t * lm ) {
    delete lm;
}

void moshi_lm_set_delay_steps( moshi_lm_t * lm, int delay_steps ) {
    lm->model->delay_steps = delay_steps;
}

int moshi_lm_get_max_delay( moshi_lm_t * lm ) {
    return lm->model->max_delay;
}

int moshi_lm_get_delay_steps( moshi_lm_t * lm ) {
    return lm->model->delay_steps;
}

int moshi_lm_get_text_card( moshi_lm_t * lm ) {
    return lm->model->text_card;
}

int moshi_lm_get_text_padding_token_id( moshi_lm_t * lm ) {
    return lm->model->text_padding_token_id;
}

int moshi_lm_get_text_new_word_token_id( moshi_lm_t * lm ) {
    (void) lm;
    return TokenIds().new_word;
}

int moshi_lm_get_text_delay( moshi_lm_t * lm ) {
    return lm->model->delays.empty() ? 0 : lm->model->delays[0];
}

bool moshi_lm_quantize( moshi_lm_t * lm, const char * quant ) {
    // parse quant string — compare first 5 bytes for mixed types, 4 for standard
    std::string q(quant);
    ggml_type qtype = (ggml_type)0;
    bool mixed = false;

    if ( q == "q3_k" ) {
        qtype = GGML_TYPE_Q3_K;
    } else if ( q == "q4_0" ) {
        qtype = GGML_TYPE_Q4_0;
    } else if ( q == "q4_k" ) {
        qtype = GGML_TYPE_Q4_K;
    } else if ( q == "q5_k" ) {
        qtype = GGML_TYPE_Q5_K;
    } else if ( q == "q6_k" ) {
        qtype = GGML_TYPE_Q6_K;
    } else if ( q == "q8_0" ) {
        qtype = GGML_TYPE_Q8_0;
    } else if ( q == "q3_k_m" ) {
        // mixed: transformer linear layers at Q3_K, embeddings at Q5_K
        qtype = GGML_TYPE_Q3_K;
        mixed = true;
    } else if ( q == "q4_k_m" ) {
        // mixed: transformer linear layers at Q4_K, embeddings at Q8_0
        qtype = GGML_TYPE_Q4_K;
        mixed = true;
    } else if ( q == "q5_k_m" ) {
        // mixed: transformer linear layers at Q5_K, embeddings at Q8_0
        qtype = GGML_TYPE_Q5_K;
        mixed = true;
    } else {
        return false;
    }

    lm->weights->quantize = true;
    lm->weights->qtype = qtype;
    lm->weights->quantize_mixed = mixed;

    // set fallback type for sensitive layers in mixed mode
    if ( mixed ) {
        if ( qtype == GGML_TYPE_Q3_K ) {
            lm->weights->qtype_fallback = GGML_TYPE_Q5_K;
        } else {
            lm->weights->qtype_fallback = GGML_TYPE_Q8_0;
        }
    }
    return true;
}

int moshi_lm_load( moshi_lm_t * lm ) {
    if ( lm->weights->is_gguf ) {
        lm->weights->load_gguf( );
    }

    get_weights( lm->weights, "lm.", lm->model );
    if ( lm->uses_cross ) {
        lm->cond = new conditioners_t;
        get_weights( lm->weights, lm->cond );
    }

    if ( ! lm->weights->is_gguf ) {
        lm->weights->load();
    }

    return 0;
}

void moshi_lm_save_gguf( moshi_lm_t * lm, const char * filepath ) {
    lm->weights->save_gguf( filepath );
}


// MARK: Generator

struct moshi_lm_gen_t {
    moshi_lm_t * lm;

    own_ptr<voice_t> voice;
    own_ptr<WeightLoader> voice_weights;

    moshi_lmgen_t lmgen;
    // Owned by this generator, all six, and all allocated in moshi_lm_start.
    //
    // The NSDMIs are load-bearing, not tidiness: moshi_lm_generator does
    // `new moshi_lm_gen_t` (default-init), so without them these are indeterminate
    // between the generator being created and moshi_lm_start filling them in --
    // and unref() now dereferences them. A caller that allocates a generator and
    // then fails before starting it is an ordinary path, not a bug.
    StateMachine * machine = NULL;
    State * machine_state = NULL;
    StateContext * state_ctx = NULL;
    ScratchContext * ctx = NULL;
    std::vector<int> audio_tokens;

    moshi_lmmodel_states_t * lm_states = NULL;
    moshi_lmgen_state_t * lmgen_state = NULL;

    // One-shot injection slot handed to moshi_lmgen_t on the personaplex path.
    // -1 == nothing armed. Atomic because it is armed from the caller's thread and
    // consumed on the inference thread. NSDMI because moshi_lm_generator
    // default-initializes.
    std::atomic<int> personaplex_forced_text_token{-1};
};

moshi_lm_gen_t * moshi_lm_generator( moshi_lm_t * lm ) {
    auto gen = new moshi_lm_gen_t;
    gen->lm = lm;
    return gen;
}

void unref( moshi_lm_gen_t * gen ) {
    if ( ! gen )
        return;
    // `delete gen` alone leaks every one of these: they are raw pointers to
    // heap objects this generator owns, and two of them -- state_ctx and ctx --
    // each hold a ggml_backend_buffer_t of DEVICE memory. Their destructors are
    // correct and were simply never reached.
    //
    // Measured before this change, one PersonaPlex-7B bf16 session on a 20 GiB
    // gfx1100: idle 25 MiB -> session resident 18987 MiB -> 3481 MiB fifteen
    // seconds after teardown. 3.46 GiB stranded per session, so the second
    // session in a process cannot allocate and ggml answers that with
    // GGML_ASSERT -> abort().
    //
    // Reverse allocation order: lm_states holds tensors living in state_ctx, so
    // state_ctx goes last. Nothing here reads another's memory during teardown --
    // the order is defensive, not required.
    delete gen->lmgen_state;
    delete gen->lm_states;
    delete gen->machine_state;
    delete gen->machine;
    delete gen->ctx;
    delete gen->state_ctx;
    delete gen;
}

int moshi_lm_set_voice_condition( moshi_context_t * moshi, moshi_lm_gen_t * gen, const char * filepath ) {
    if ( ! gen->lm->uses_cross )
        return -1;

    gen->voice_weights = WeightLoader::from_safetensor( filepath, moshi->scratch_cpu, moshi->backend );
    if ( ! gen->voice_weights )
        return -1;

    return 0;
}

int moshi_lm_load_voice_condition( moshi_context_t * moshi, moshi_lm_gen_t * gen ) {
    if ( ! gen->lm->uses_cross )
        return -1;

    if ( ! gen->lm->cond )
        return -2;

    gen->voice = new voice_t;
    gen->voice->ctx = NULL;
    gen->voice->buffer = NULL;
    gen->voice->sum = NULL;
    gen->voice->cross = NULL;

    ggml_tensor * speaker_wavs;
    gen->voice_weights->fetch( &speaker_wavs, "voice.speaker_wavs" );
    gen->voice_weights->load();

    voice_condition( gen->voice, gen->lm->cond, speaker_wavs, moshi );

    return 0;
}

int moshi_lm_voice_prefix( moshi_lm_gen_t * gen, std::deque<int> & text_prefix, std::deque<std::vector<int>> & audio_prefix ) {
    gen->voice = new voice_t;
    gen->voice->ctx = NULL;
    gen->voice->buffer = NULL;
    gen->voice->sum = NULL;
    gen->voice->cross = NULL;
    gen->voice->text_prefixes.swap( text_prefix );
    gen->voice->audio_prefixes.swap( audio_prefix );
    return 0;
}

int moshi_lm_personaplex_load_voice( moshi_context_t * moshi, moshi_lm_gen_t * gen, const char * filepath ) {
    std::string filename = filepath;
    auto ext_index = filename.find_last_of('.');
    if ( ext_index == std::string::npos ) {
        return -1;
    }
    std::string ext = filename.substr(ext_index);
    if ( ext == ".safetensors" ) {
        gen->voice_weights = WeightLoader::from_safetensor( filepath,
            moshi->scratch_cpu, moshi->backend );
        if ( ! gen->voice_weights )
            return -1;
    } else if ( ext == ".gguf" ) {
        gen->voice_weights = WeightLoader::from_gguf( filepath,
            moshi->scratch_cpu, moshi->backend );
        if ( ! gen->voice_weights )
            return -1;

        gen->voice_weights->load_gguf();
    } else {
        return -1;
    }

    int n;
    ggml_tensor * voice_prompt_embeddings, * voice_prompt_cache;
    n = gen->voice_weights->fetch( &voice_prompt_embeddings, "voice.embeddings" );
    assert( n );
    n = gen->voice_weights->fetch( &voice_prompt_cache, "voice.cache" );
    assert( n );
    if ( ! gen->voice_weights->is_gguf ) {
        gen->voice_weights->load();
//#define SAVE_GGUF
#ifdef SAVE_GGUF
        gen->voice_weights->save_gguf( (filename.substr(0, ext_index) + ".gguf").c_str() );
#endif
    }

    gen->voice = new voice_t;
    gen->voice->ctx = NULL;
    gen->voice->buffer = NULL;
    gen->voice->sum = NULL;
    gen->voice->cross = NULL;
    gen->voice->prompt_embeddings = voice_prompt_embeddings;
    gen->voice->prompt_cache = voice_prompt_cache;

    return 0;
}

static voice_t * moshi_lm_personaplex_voice( moshi_lm_gen_t * gen ) {
    if ( ! gen->voice ) {
        gen->voice = new voice_t{};
        gen->voice->ctx = NULL;
        gen->voice->buffer = NULL;
        gen->voice->sum = NULL;
        gen->voice->cross = NULL;
        gen->voice->prompt_embeddings = NULL;
        gen->voice->prompt_cache = NULL;
    }
    return gen->voice;
}

void moshi_lm_personaplex_set_text_prompt( moshi_lm_gen_t * gen, tokenizer_t * tok, const char * text ) {
    auto voice = moshi_lm_personaplex_voice( gen );
    voice->text_prompt_tokens.clear();
    if ( ! text || ! text[0] )
        return;

    // Encode the WHOLE prompt in one SentencePiece call.
    //
    // This used to split on whitespace and Encode() each word separately, which
    // is not the same tokenization: SentencePiece prefixes a word-initial marker
    // (U+2581 '_') per Encode() call, so "a b" encoded whole and encoded as two
    // words do not agree. The reference PersonaPlex server does the single whole
    // string call (text_tokenizer.encode(prompt)), and the system prefix has to
    // match it token for token or the model is conditioned on something else.
    tok->sp.Encode( std::string( text ), &voice->text_prompt_tokens );
    printf("text prompt: %d tokens\n", (int)voice->text_prompt_tokens.size());
}

void moshi_lm_personaplex_set_text_prompt_tokens( moshi_lm_gen_t * gen, const int * tokens, int n_tokens ) {
    auto voice = moshi_lm_personaplex_voice( gen );
    voice->text_prompt_tokens.clear();
    if ( ! tokens || n_tokens <= 0 )
        return;
    voice->text_prompt_tokens.assign( tokens, tokens + n_tokens );
}

int moshi_lm_personaplex_get_text_prompt_tokens( moshi_lm_gen_t * gen, std::vector<int> & tokens ) {
    tokens.clear();
    if ( ! gen->voice )
        return 0;
    tokens = gen->voice->text_prompt_tokens;
    return (int) tokens.size();
}

void moshi_lm_prime( moshi_context_t * moshi, moshi_lm_gen_t * gen, int audio_silence_frames );

void moshi_lm_start( moshi_context_t * moshi, moshi_lm_gen_t * gen, float depth_temperature, float text_temperature, bool logging, int audio_silence_frames, bool prime ) {
    const int max_padding = 8;
    const int initial_padding = 2;
    const int second_stream_ahead = gen->lm->second_stream_ahead;
    gen->state_ctx = new StateContext( moshi->backend );
    ggml_tensor * condition_cross = NULL;
    if ( gen->voice && ! gen->lm->model->personaplex ) {
        condition_cross = gen->voice->cross;
        gen->machine = new StateMachine(gen->lm->model->text_card + 1, second_stream_ahead, max_padding, initial_padding,
            gen->lm->model->text_padding_token_id);
        gen->machine->logging = logging;
        gen->machine_state = gen->machine->new_state();
        gen->lmgen = moshi_lmgen_t{
            gen->lm->model,
            true, depth_temperature, text_temperature, 250, 25,
            gen->machine, gen->machine_state,
            gen->voice->sum,
            &gen->voice->text_prefixes, &gen->voice->audio_prefixes,
            NULL // no injection on the state-machine (TTS) path
        };
        gen->lm_states = moshi_lmmodel_states( gen->state_ctx, gen->lm->model, gen->voice->cross );
    } else {
        gen->lmgen = moshi_lmgen_t{
            gen->lm->model,
            true, depth_temperature, text_temperature, 250, 25,
            NULL, NULL, // no state machine
            NULL, // no cross
            NULL, NULL, // empty prefixes
            &gen->personaplex_forced_text_token,
        };
        gen->personaplex_forced_text_token.store( -1, std::memory_order_release );
        gen->lm_states = moshi_lmmodel_states( gen->state_ctx, gen->lm->model, NULL );
    }
    gen->lmgen_state = moshi_lmgen_state( gen->lm->model );
    gen->state_ctx->alloc();

    gen->ctx = new ScratchContext( 256, moshi->backend );
    gen->audio_tokens.resize( gen->lm->model->num_audio_codebooks );

    // prime=false stops HERE, with the arena allocated and its device tensors
    // still uninitialized. That is deliberate and it is not a half-measure: the
    // only sanctioned next step is a snapshot restore, which writes every one of
    // those tensors plus every scalar the seed would have set. Seeding first would
    // be a 1.5 GiB host-to-device pass whose every byte the restore then
    // overwrites -- measured at 1.8 s on this card, which is most of what the
    // snapshot exists to save.
    if ( prime )
        moshi_lm_prime( moshi, gen, audio_silence_frames );
}

// The PRIME half of moshi_lm_start: put the generator back to the state a fresh
// conversation starts from, touching only things that already exist.
//
// WHY THIS IS SPLIT OUT -- REUSE OVER FREE. A server that holds a model resident
// and serves one conversation at a time used to destroy the generator between
// connections and build a new one. That is a free/alloc cycle of the KV cache, the
// state context and the compiled compute graphs on every reconnect, and on a card
// sized for exactly one session it does not survive the second one: the freed
// blocks come back fragmented and a ~794 MiB graph allocation fails, which ggml
// answers with GGML_ASSERT -> abort().
//
// Resetting instead allocates nothing and frees nothing. The arena is paid for once
// and reused for the life of the process, so the per-connection memory delta is
// zero by construction rather than by getting every destructor right.
//
// Everything conversational is re-seeded here: the delay-cache ring and its
// `provided` masks, the offset, the state tensors (state_ctx->init() rewrites them
// from the saved initial data), the model's streaming states, the one-shot
// injection slot, and finally the system-prompt phase -- which picks up whatever
// voice and text prompt the caller has set since, so a new conversation gets its
// own persona rather than the previous one's.
void moshi_lm_prime( moshi_context_t * moshi, moshi_lm_gen_t * gen, int audio_silence_frames ) {
    ggml_tensor * condition_cross =
        ( gen->voice && ! gen->lm->model->personaplex ) ? gen->voice->cross : NULL;

    gen->personaplex_forced_text_token.store( -1, std::memory_order_release );
    moshi_lmgen_state_seed( gen->lmgen_state, gen->lm->model );
    gen->state_ctx->init();
    init( moshi->scratch, gen->lm_states, gen->lm->model, condition_cross );

    // personaplex process system prompts
    if ( gen->lm->model->personaplex ) {
        moshi_lmgen_step_system_prompts(
            *gen->ctx,
            &gen->lmgen, gen->lmgen_state,
            gen->lm_states,
            gen->voice,
            audio_silence_frames
        );
    }
}

void moshi_lm_reset( moshi_context_t * moshi, moshi_lm_gen_t * gen, int audio_silence_frames ) {
    // Only meaningful on a generator moshi_lm_start has already built.
    if ( ! gen || ! gen->state_ctx || ! gen->lm_states || ! gen->lmgen_state || ! gen->ctx )
        return;
    moshi_lm_prime( moshi, gen, audio_silence_frames );
}

// MARK: Primed-state snapshot
//
// WHAT IS IN HERE IS THE WHOLE CONTRACT, so it is worth naming each piece and why
// the ones that are absent are absent.
//
//   * `tensors` -- every tensor the state context owns, in registration order:
//     the transformer and depformer KV caches, any cached cross-attention k/v, and
//     lm_states->transformer_out. This is the bulk, and it is bounded by the STATE
//     context, not by the weights.
//   * `transformer_offset` -- the main transformer's live offset. It indexes the KV
//     ring on every step (moshi_streaming_transformer_graph_step reads it and
//     advances it), so a snapshot that restored the cache but not this one would
//     read the right memory at the wrong positions: consistent, plausible, wrong.
//   * `lmgen` -- the generator's delay cache, its `provided` masks, the offset and
//     the initial tokens. The prompt phase leaves the cache holding the restored
//     voice cache and the masks holding the last replayed step's flags, and the
//     first conversation step reads both.
//
// DELIBERATELY ABSENT:
//   * The depformer's offset. It is consumed ONLY while the depformer graph is
//     being built -- moshi_lmmodel_depformer_step builds once and thereafter only
//     computes -- so after the build it is inert, and before the build the only
//     correct value is the 0 that init() leaves. Restoring the captured value into
//     a generator that has not built its depformer graph yet would bake the wrong
//     cache indices into it, which is why restore writes 0 rather than the capture.
//   * The compiled graphs (lm_states->gctx, depformer_gctx, and the transformer's
//     graph / lazy_graph / conv_graph slots). Those are compute plans, not
//     conversation state; they are built on demand and each discipline restores its
//     own view before stepping (see the comments in transformer.h).
//   * The voice and the text prompt. They are INPUTS to the state this snapshot
//     holds, not part of it -- see the header's note that keying is the caller's.
//   * StateMachine state, which is why both entry points refuse a generator that
//     has one.
struct moshi_lm_snapshot_t {
    // Identity: which model's state layout this blob describes.
    const moshi_lm_t * lm = NULL;
    size_t n_tensors = 0;

    std::vector<uint8_t> tensors;
    int transformer_offset = 0;
    moshi_lmgen_state_t lmgen;

    bool valid = false;
};

moshi_lm_snapshot_t * moshi_lm_snapshot_alloc() {
    return new moshi_lm_snapshot_t;
}

void unref( moshi_lm_snapshot_t * snapshot ) {
    delete snapshot;
}

size_t moshi_lm_snapshot_bytes( moshi_lm_snapshot_t * snapshot ) {
    if ( ! snapshot || ! snapshot->valid )
        return 0;
    size_t bytes = snapshot->tensors.size();
    // The host-side halves, counted honestly: a caller sizing a cache wants the
    // real per-entry cost, and the delay cache is not free.
    for ( const auto & slot : snapshot->lmgen.cache )
        bytes += slot.size() * sizeof( int );
    for ( const auto & slot : snapshot->lmgen.provided )
        bytes += slot.size();
    bytes += snapshot->lmgen.initial.size() * sizeof( int );
    return bytes;
}

size_t moshi_lm_snapshot_state_bytes( moshi_lm_gen_t * gen ) {
    if ( ! gen || ! gen->state_ctx )
        return 0;
    return gen->state_ctx->state_nbytes();
}

int moshi_lm_snapshot_capture( moshi_lm_gen_t * gen, moshi_lm_snapshot_t * snapshot ) {
    if ( ! gen || ! snapshot )
        return -1;
    if ( ! gen->state_ctx || ! gen->lm_states || ! gen->lmgen_state )
        return -1;
    if ( gen->machine )
        return -2;

    snapshot->valid = false;
    if ( ! gen->state_ctx->save( snapshot->tensors ) )
        return -1;
    snapshot->lm                 = gen->lm;
    snapshot->n_tensors          = gen->state_ctx->states.size();
    snapshot->transformer_offset = gen->lm_states->transformer->offset;
    snapshot->lmgen              = *gen->lmgen_state;
    snapshot->valid              = true;
    return 0;
}

int moshi_lm_snapshot_restore( moshi_lm_gen_t * gen, moshi_lm_snapshot_t * snapshot ) {
    if ( ! gen || ! snapshot || ! snapshot->valid )
        return -1;
    if ( ! gen->state_ctx || ! gen->lm_states || ! gen->lmgen_state )
        return -1;
    if ( gen->machine )
        return -2;
    // Same model and same registration list, or the blob describes something else.
    if ( snapshot->lm != gen->lm )
        return -3;
    if ( snapshot->n_tensors != gen->state_ctx->states.size() )
        return -3;
    if ( ! gen->state_ctx->load( snapshot->tensors ) )
        return -3;

    gen->lm_states->transformer->offset = snapshot->transformer_offset;
    if ( gen->lm_states->depformer )
        gen->lm_states->depformer->offset = 0;   // see the comment on the struct
    *gen->lmgen_state = snapshot->lmgen;
    gen->personaplex_forced_text_token.store( -1, std::memory_order_release );
    return 0;
}

int moshi_lm_get_text_logits( moshi_lm_gen_t * gen, std::vector<float> & logits ) {
    logits.clear();
    if ( ! gen->lm_states || ! gen->lm_states->text_logits_out )
        return 0;
    auto tensor = gen->lm_states->text_logits_out;
    logits.resize( (size_t) ggml_nelements( tensor ) );
    ggml_backend_tensor_get( tensor, logits.data(), 0, ggml_nbytes( tensor ) );
    return (int) logits.size();
}

void moshi_lm_personaplex_force_text_token( moshi_lm_gen_t * gen, int token ) {
    gen->personaplex_forced_text_token.store( token, std::memory_order_release );
}

void moshi_lm_personaplex_clear_forced_text_token( moshi_lm_gen_t * gen ) {
    gen->personaplex_forced_text_token.store( -1, std::memory_order_release );
}

int moshi_lm_personaplex_pending_forced_text_token( moshi_lm_gen_t * gen ) {
    return gen->personaplex_forced_text_token.load( std::memory_order_acquire );
}

void moshi_lm_send( moshi_lm_gen_t * gen, Entry * entry ) {
    gen->machine_state->entries.push_back( *entry );
}

int moshi_lm_receive( moshi_lm_gen_t * gen, int & text_token, std::vector<int16_t> & audio_tokens ) {
    bool depformer_replace_tokens = (gen->lmgen_state->offset < gen->lm->model->delay_steps);
    bool has_audio_tokens = moshi_lmgen_step(
        *gen->ctx,
        &gen->lmgen, gen->lmgen_state,
        gen->lm_states,
        depformer_replace_tokens,
        text_token,
        gen->audio_tokens
    );
    audio_tokens.resize( gen->audio_tokens.size() );
    if ( has_audio_tokens ) {
        for ( int i = 0; i < gen->audio_tokens.size(); ++i )
            audio_tokens[i] = gen->audio_tokens[i];
    }
    return has_audio_tokens? 1 : 0;
}

void moshi_lm_send2( moshi_lm_gen_t * gen, std::vector<int16_t> & audio_tokens ) {
    gen->audio_tokens.resize( audio_tokens.size() );
    for ( int i = 0; i < audio_tokens.size(); ++i )
        gen->audio_tokens[i] = audio_tokens[i];
}

void moshi_lm_receive2( moshi_lm_gen_t * gen, int & text_token, float & vad ) {
    moshi_lmgen_step(
        *gen->ctx,
        &gen->lmgen, gen->lmgen_state,
        gen->lm_states,
        false, //depformer_replace_tokens
        text_token,
        gen->audio_tokens,
        &vad
    );
}

int moshi_lm_is_active( moshi_lm_gen_t * gen ) {
    const int final_padding = 4;
    int end_offset = gen->machine_state->end_step + gen->lm->model->delay_steps + final_padding;
    int offset = gen->lmgen_state->offset;
    return ( offset < end_offset || gen->machine_state->end_step == -1 );
}

int moshi_lm_is_empty( moshi_lm_gen_t * gen ) {
    return gen->machine_state->is_empty();
}

void moshi_lm_machine_reset( moshi_lm_gen_t * gen ) {
    gen->machine->reset_state( gen->machine_state );
}

// MARK: Misc

/*int moshi_lm_n_q( moshi_lmmodel_t * lm ) {
    return lm->n_q;
}

int moshi_lm_max_delay( moshi_lmmodel_t * lm ) {
    return lm->max_delay;
}

int moshi_lm_delay_steps( moshi_lmmodel_t * lm ) {
    return lm->delay_steps;
}*/






