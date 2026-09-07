#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <vector>

#include <moshi/script_grammar.h>

// The padding id every caller fell back on before it was read from the model
// config. Kept as the fallback for configs that omit `existing_text_padding_id`,
// and as the default here so no caller has to spell it.
const int MOSHI_DEFAULT_TEXT_PADDING_ID = 3;

class TokenIds {
public:
    int card;
    const int new_word = 0;
    const int pad;
    const int main = 1;
    const int other = 2;
    const int zero = -1;
    const int ungenerated = -2;

    TokenIds(int card = 8001, int pad = MOSHI_DEFAULT_TEXT_PADDING_ID)
        : pad(pad) {
        this->card = card;
    }
};

class State {
public:
    int remaining_padding;
    int forced_padding;
    int end_step;
    std::deque<Entry> entries;
    std::deque<int> queued;
    std::deque<int> lookahead_queued;

    std::vector<int> get_tokens_ahead(int lookahead) {
        // assert lookahead > 0
        for (auto entry : entries) {
            if (!entry.tokens.size())
                continue;
            lookahead -= 1;
            if (lookahead != 0)
                continue;
            return entry.tokens;
        }
        return {};
    }
    
    bool is_empty() {
        if (entries.size())
            return false;
        if (queued.size())
            return false;
        if (lookahead_queued.size())
            return false;
        return true;
    }
};

int64_t g_last_token_time;

class StateMachine {
public:
    TokenIds token_ids;
    int second_stream_ahead;
    int max_padding;
    int initial_padding;
    bool logging = false;

    StateMachine(
            int text_card,
            int second_stream_ahead = 0,
            int max_padding = 6,
            int initial_padding = 2,
            int text_padding_token_id = MOSHI_DEFAULT_TEXT_PADDING_ID)
            : token_ids( text_card, text_padding_token_id ) {
        this->second_stream_ahead = second_stream_ahead;
        this->max_padding = max_padding;
        this->initial_padding = initial_padding;
    }

    State * new_state(std::deque<Entry> &entries) {
        auto state = new State(
            /*remaining_padding=*/initial_padding,
            /*forced_padding=*/initial_padding,
            /*end_step*/-1,
            entries
        );
        return state;
    }

    State * new_state() {
        auto state = new State(
            /*remaining_padding=*/initial_padding,
            /*forced_padding=*/initial_padding,
            /*end_step*/-1
        );
        return state;
    }
    
    void reset_state( State * state ) {
        state->remaining_padding = initial_padding;
        state->forced_padding = initial_padding;
        state->end_step = -1;
        state->entries.clear();
        state->queued.clear();
        state->lookahead_queued.clear();
    }

    int process(int step, State * state, int token) {

        if (token != token_ids.new_word && token != token_ids.pad)
            token = token_ids.pad;

        if (state->queued.size())
            // Some text tokens are yet to be fed, we must PAD.
            token = token_ids.pad;
        else if (state->forced_padding > 0)
            // We are forced to pad, we must PAD.
            token = token_ids.pad;
        else if (state->remaining_padding <= 0)
            // We are not allowed to pad, we must ask for a new WORD.
            token = token_ids.new_word;

        if (token == token_ids.new_word) {
            if (state->entries.size()) {
                auto entry = state->entries.front();
                state->entries.pop_front();

                if ( logging ) {
                    auto new_token_time = ggml_time_ms();
                    auto last_token_time = g_last_token_time;
                    if ( entry.time != 0 && entry.time > last_token_time )
                        last_token_time = entry.time;
                    printf("\"%s\" %.4f\n", entry.text.c_str(), (new_token_time - last_token_time) / 1000.f);
                    g_last_token_time = new_token_time;
                }

                if (entry.tokens.size()) {
                    // We queue the tokens to be fed to the model.
                    for (auto token : entry.tokens)
                        state->queued.push_back(token);
                    if (second_stream_ahead) {
                        // We queue the tokens for the N+lookahead word into the second text stream.
                        for (auto token : state->get_tokens_ahead(second_stream_ahead))
                            state->lookahead_queued.push_back(token);
                    }
                    // Entry contains a new word, we reset the max padding counter.
                    state->remaining_padding = max_padding;
                } else {
                    token = token_ids.pad;
                }
                state->forced_padding = entry.padding;
            } else {
                token = token_ids.pad;
                if (second_stream_ahead && state->end_step < 0)
                    token = token_ids.new_word;
                // Trying to consume past the last word, we reached the end.
                if (state->end_step < 0)
                    state->end_step = step;
            }
        }

        int output = 0;
        if (token == token_ids.pad) {
            // Decrement the counters for remaining and forced pads.
            if (state->remaining_padding > 0)
                state->remaining_padding -= 1;
            if (state->forced_padding > 0)
                state->forced_padding -= 1;
            if (state->queued.size()){
                // We have some text tokens to feed to the model.
                output = state->queued.front();
                state->queued.pop_front();
            } else {
                output = token_ids.pad;
            }
        } else if (token == token_ids.new_word) {
            output = token_ids.new_word;
        } else if (token == token_ids.zero) {
            output = token;
        }

        if (second_stream_ahead) {
            int second = -1;
            if (output == token_ids.new_word) {
                second = token_ids.new_word;
                if (state->queued.size()) {
                    output = state->queued.front();
                    state->queued.pop_front();
                } else {
                    output = token_ids.pad;
                }
            } else if (state->lookahead_queued.size()) {
                second = state->lookahead_queued.front();
                state->lookahead_queued.pop_front();
            }
            output = (second + 1) * token_ids.card + output;
        }
        return output;
    }
};

template<class T>
void script_to_entries(
    std::deque<Entry> &entries,
    T &tokenizer,
    TokenIds &token_ids,
    float frame_rate,
    std::vector<std::string> script,
    bool multi_speaker = true,
    int padding_between = 0
) {
    int speaker_tokens[] = {token_ids.main, token_ids.other};
    int last_speaker = -99;
    for (size_t idx = 0; idx < script.size(); idx++) {
        bool first_content = true;
        auto init_line = script[idx];
        std::string line;
        for (size_t i = 0; i < init_line.size(); i++) {
            char c = init_line[i];
            switch(c) {
            //case '’': line += '\''; break;
            case ':': line += ' '; break;
            case '(': break;
            case ')': break;
            default: line += c;
            }
        }
        // TODO: experiment with break
        // break is indicated as e.g. <break time="3s"/>
        // event_re = re.compile(r"(?:<break\s+time=\"([0-9]+(?:.[0-9]*)?)s\"\s*/?>)|(?:\s+)")
        std::string ws = " \t\r\n";
        auto cur = line.find_first_not_of(ws);
        while (cur != std::string::npos) {
            auto end = line.find_first_of(ws, cur);
            std::string word;
            if (end == std::string::npos) {
                word = line.substr(cur);
                cur = std::string::npos;
            } else {
                auto count = end - cur;
                word = line.substr(cur, count);
                cur = line.find_first_not_of(ws, end);
            }
            std::vector<int> tokens;
            tokenizer.Encode(word, &tokens);
            if (first_content) {
                int speaker = idx % 2; // len(speaker_tokens)
                if (multi_speaker && last_speaker != speaker) {
                    last_speaker = speaker;
                    std::vector<int> new_tokens(1 + tokens.size());
                    new_tokens[0] = speaker_tokens[speaker];
                    for (size_t i = 0; i < tokens.size(); i++)
                        new_tokens[i+1] = tokens[i];
                    tokens.swap(new_tokens);
                }
                first_content = false;
            }
            int padding = 0;
            if (padding_between > 0) {
                padding = padding_between + tokens.size() - 1;
                if (padding < 0) padding = 0;
            }
            entries.push_back(Entry(tokens, word, padding));
        }
    }
}

template<class T>
void script_to_state(
    State * state,
    T &tokenizer,
    TokenIds &token_ids,
    float frame_rate,
    std::vector<std::string> script,
    bool multi_speaker = true,
    int padding_between = 0
) {
    int speaker_tokens[] = {token_ids.main, token_ids.other};
    int last_speaker = -99;
    for (size_t idx = 0; idx < script.size(); idx++) {
        bool first_content = true;
        auto init_line = script[idx];
        std::string line;
        for (size_t i = 0; i < init_line.size(); i++) {
            char c = init_line[i];
            switch(c) {
            //case '’': line += '\''; break;
            case ':': line += ' '; break;
            case '(': break;
            case ')': break;
            default: line += c;
            }
        }
        // TODO: experiment with break
        // break is indicated as e.g. <break time="3s"/>
        // event_re = re.compile(r"(?:<break\s+time=\"([0-9]+(?:.[0-9]*)?)s\"\s*/?>)|(?:\s+)")
        std::string ws = " \t\r\n";
        auto cur = line.find_first_not_of(ws);
        while (cur != std::string::npos) {
            auto end = line.find_first_of(ws, cur);
            std::string word;
            if (end == std::string::npos) {
                word = line.substr(cur);
                cur = std::string::npos;
            } else {
                auto count = end - cur;
                word = line.substr(cur, count);
                cur = line.find_first_not_of(ws, end);
            }
            std::vector<int> tokens;
            tokenizer.Encode(word, &tokens);
            if (first_content) {
                int speaker = idx % 2; // len(speaker_tokens)
                if (multi_speaker && last_speaker != speaker) {
                    last_speaker = speaker;
                    std::vector<int> new_tokens(1 + tokens.size());
                    new_tokens[0] = speaker_tokens[speaker];
                    for (size_t i = 0; i < tokens.size(); i++)
                        new_tokens[i+1] = tokens[i];
                    tokens.swap(new_tokens);
                }
                first_content = false;
            }
            int padding = 0;
            if (padding_between > 0) {
                padding = padding_between + (int) tokens.size() - 1;
                if (padding < 0) padding = 0;
            }
            state->entries.push_back(Entry(tokens, word, padding));
        }
    }
}

/*************************************************************\
 *  moshi.models.LMModel
\*************************************************************/

struct moshi_lmmodel_t {
    int n_q;
    int dep_q;
    int card;
    int text_card;
    std::vector<int> delays;
    int max_delay;
    int dim;
    std::vector<int> depformer_weights_per_step_schedule;
    own_ptr_vector<moshi_scaled_embedding_t> emb;
    bool demux_second_stream;
    own_ptr<moshi_scaled_embedding_demux_t> text_emb_demux;
    own_ptr<moshi_scaled_embedding_t> text_emb;

    own_ptr<torch_nn_linear_t> text_linear;
    own_ptr<moshi_streaming_transformer_t> transformer;
    own_ptr<moshi_rms_norm_t> out_norm;
    bool depformer_multi_linear;

    own_ptr_vector<torch_nn_linear_t> depformer_in;
    own_ptr_vector<moshi_scaled_embedding_t> depformer_emb;
    own_ptr<moshi_scaled_embedding_demux_t> depformer_text_emb_demux;
    own_ptr<moshi_scaled_embedding_t> depformer_text_emb;
    own_ptr<moshi_streaming_transformer_t> depformer;

    own_ptr_vector<torch_nn_linear_t> extra_heads;

    own_ptr_vector<torch_nn_linear_t> linears;

    int num_codebooks; // n_q + 1
    int num_audio_codebooks; // n_q
    int audio_offset; // 1
    int delay_steps;
    int text_initial_token_id;
    int initial_token_id;
    // From the model config's `existing_text_padding_id`. Every site that used to
    // spell 3 reads this instead.
    int text_padding_token_id;
    bool personaplex;
};

void get_weights( WeightLoader * loader, std::string path, moshi_lmmodel_t * lm ) {
    for ( size_t i = 0; i < lm->depformer_in.size(); i++ )
        get_weights( loader, path + "depformer_in."+std::to_string(i)+".", lm->depformer_in[i] );
    if ( lm->depformer ) {
        get_weights( loader, path + "depformer.", lm->depformer );
        if ( lm->demux_second_stream )
            get_weights( loader, path + "depformer_text_emb.", lm->depformer_text_emb_demux );
        else
            get_weights( loader, path + "depformer_text_emb.", lm->depformer_text_emb );
        for ( size_t i = 0; i < lm->depformer_emb.size(); i++ )
            get_weights( loader, path + "depformer_emb."+std::to_string(i)+".", lm->depformer_emb[i] );
    }
    for ( size_t i = 0; i < lm->extra_heads.size(); i++ )
        get_weights( loader, path + "extra_heads."+std::to_string(i)+".", lm->extra_heads[i] );
    for ( size_t i = 0; i < lm->linears.size(); i++ )
        get_weights( loader, path + "linears."+std::to_string(i)+".", lm->linears[i] );
    for ( size_t i = 0; i < lm->emb.size(); i++ )
        get_weights( loader, path + "emb."+std::to_string(i)+".", lm->emb[i] );
    if ( lm->demux_second_stream )
        get_weights( loader, path + "text_emb.", lm->text_emb_demux );
    else
        get_weights( loader, path + "text_emb.", lm->text_emb );
    get_weights( loader, path + "transformer.", lm->transformer);
    get_weights( loader, path + "out_norm.", lm->out_norm);
    get_weights( loader, path + "text_linear.", lm->text_linear);
}

struct lmmodel_embed_t {
    embedding_demux_t text_demux;
    embedding_t text;
    std::vector<embedding_t> audio;
};

struct lmmodel_depformer_embed_t {
    embedding_demux_t text_demux;
    embedding_t text;
};

struct moshi_lmmodel_states_t {
    own_ptr<moshi_streaming_transformer_state_t> transformer;
    own_ptr<moshi_streaming_transformer_state_t> depformer;

    // The conversation path's own copy of transformer->graph, restored before every
    // step. Counterpart to moshi_streaming_transformer_state_t::lazy_graph: the two
    // disciplines share one live slot and each keeps its own view. See the comment
    // there for what went wrong without it.
    moshi_streaming_transformer_graph_t conv_graph;

    GraphContext * gctx = NULL;
    lmmodel_embed_t embed;
    int  transformer_T;
    ggml_tensor * transformer_out;
    ggml_tensor * sampler_out;
    // The text logits the sampler consumed, kept reachable so a caller can read the
    // DISTRIBUTION and not just the argmax. moshi_lm_get_text_logits() is the public
    // door; see there for why a parity check needs it.
    ggml_tensor * text_logits_out = NULL;

    GraphContext * depformer_gctx = NULL;
    lmmodel_depformer_embed_t depformer_embed;
    ggml_tensor * depformer_tokens;

    // Both GraphContexts are owned here and each holds a ggml_backend_buffer_t.
    // ~GraphContext frees it correctly; nothing was calling it, so every
    // generator teardown stranded the two compute buffers along with the state.
    ~moshi_lmmodel_states_t() {
        delete depformer_gctx;
        delete gctx;
    }
};

moshi_lmmodel_states_t * moshi_lmmodel_states( StateContext * state_ctx,
        moshi_lmmodel_t * lm, ggml_tensor * k_cross ) {
    auto state = new moshi_lmmodel_states_t;
    state->transformer = moshi_streaming_transformer_state( state_ctx, lm->transformer,
        k_cross );
    if ( lm->depformer ) {
        state->depformer = moshi_streaming_transformer_state( state_ctx, lm->depformer,
            NULL );
    } else {
        state->depformer = NULL;
    }
    state_ctx->fill(GGML_NE(lm->dim), 0.f, &state->transformer_out );
    return state;
}

void init( ScratchContext * ctx, moshi_lmmodel_states_t * state,
        moshi_lmmodel_t * lm,
        ggml_tensor * condition_cross ) {
    init( ctx, state->transformer, lm->transformer, condition_cross );
    if ( state->depformer )
        init( ctx, state->depformer, lm->depformer, NULL );
}

ggml_tensor * moshi_lmmodel_forward_depformer_transform(
        GraphContext & ctx,
        moshi_lmmodel_t * lm,
        moshi_lmmodel_states_t * states,
        int depformer_cb_index,
        ggml_tensor * last_token_input,
        ggml_tensor * transformer_out
    ) {
    //ProfileScope profile(time_depformer_us);

    auto depformer_input = transformer_out;
    int in_index = 0;
    if ( lm->depformer_multi_linear ) {
        in_index = depformer_cb_index;
        if ( lm->depformer_weights_per_step_schedule.size() )
            in_index = lm->depformer_weights_per_step_schedule[in_index];
    }

    depformer_input = torch_nn_linear( ctx, lm->depformer_in[in_index], depformer_input );

    last_token_input = ggml_cast( ctx, last_token_input, GGML_TYPE_F32 );
    depformer_input = ggml_add( ctx, depformer_input, last_token_input );

    auto dep_output = moshi_streaming_transformer( ctx,
        lm->depformer, states->depformer, depformer_input );

    auto logits = torch_nn_linear( ctx, lm->linears[depformer_cb_index], dep_output );

    return logits;
}


void moshi_lmmodel_depformer_step(
        ScratchContext & scratch,
        moshi_lmmodel_t * lm,
        moshi_lmmodel_states_t * state,
        int text_token,
        bool use_sampling,
        float temp,
        int top_k,
        std::vector<int> & depformer_tokens
    ) {

    if ( ! state->depformer_gctx )
    {
        state->depformer_gctx = new GraphContext( 256, scratch.backend );
        GraphContext &ctx = *state->depformer_gctx;
        
        ggml_tensor * last_token_input;
        if ( lm->demux_second_stream ) {
            last_token_input = moshi_scaled_embedding_demux_build( ctx,
                lm->depformer_text_emb_demux, &state->depformer_embed.text_demux );
        } else {
            last_token_input = moshi_scaled_embedding_build( ctx,
                lm->depformer_text_emb, &state->depformer_embed.text );
        }

        ggml_tensor * tokens = ctx.new_tensor( GGML_TYPE_I32, GGML_NE( lm->dep_q ) );

        auto logits = moshi_lmmodel_forward_depformer_transform( ctx, lm, state,
            0, last_token_input, state->transformer_out );

        auto next_token = moshi_sample_token( ctx, logits, use_sampling, temp, top_k );
        auto view = ggml_view_1d( ctx, tokens, 1, 0 );
        ctx.build_forward_expand( ggml_cpy( ctx, next_token, view ) );

        for (int cb_index = 1; cb_index < lm->dep_q; cb_index++) {
            auto index = cb_index - 1;

            last_token_input = moshi_scaled_embedding_chained( ctx,
                lm->depformer_emb[index], next_token );

            logits = moshi_lmmodel_forward_depformer_transform(
                ctx, lm, state, cb_index,
                last_token_input,
                state->transformer_out );

            // NOT teacher-forced, unlike the reference's graphed_depth, which takes
            // target_/provided_ and substitutes a supplied token for the sampled one
            // at each codebook. That is safe HERE and only here: this chain runs
            // codebooks in order, and on personaplex every supplied codebook (the
            // user's eight) sits above every generated one (moshi's eight), so no
            // supplied value would ever have entered the chain before the generated
            // ones were produced. A model whose supplied codebooks interleaved with
            // its generated ones would need this forced, and the graph would have to
            // grow a per-codebook input slot to allow it.
            next_token = moshi_sample_token( ctx, logits, use_sampling, temp, top_k );
            view = ggml_view_1d( ctx, view, 1, 4 );
            ctx.build_forward_expand( ggml_cpy( ctx, next_token, view ) );
        }
        state->depformer_tokens = ggml_view_1d( ctx, view, tokens->ne[0], -(lm->dep_q - 1) * 4 );
        ctx.build_forward_expand( state->depformer_tokens );

        ctx.alloc();
    }
    GraphContext &ctx = *state->depformer_gctx;

    if ( lm->demux_second_stream ) {
        moshi_scaled_embedding_demux_step( ctx,
            lm->depformer_text_emb_demux,
            &state->depformer_embed.text_demux,
            text_token );
    } else {
        moshi_scaled_embedding_step( ctx,
            lm->depformer_text_emb,
            &state->depformer_embed.text,
            text_token );
    }

    ctx.compute();

    depformer_tokens.resize( ggml_nelements( state->depformer_tokens ) );
    ggml_backend_tensor_get(
        state->depformer_tokens,
        depformer_tokens.data(),
        0, ggml_nbytes( state->depformer_tokens ) );
}

ggml_tensor * moshi_lmmodel_text_token_embed_build(
        GraphContext & ctx,
        moshi_lmmodel_t * lm,
        lmmodel_embed_t * embed,
        ggml_tensor * sum_condition
    ) {

    ggml_tensor * input;
    if ( lm->demux_second_stream ) {
        input = moshi_scaled_embedding_demux_build( ctx,
            lm->text_emb_demux, &embed->text_demux );
    } else {
        input = moshi_scaled_embedding_build( ctx,
            lm->text_emb, &embed->text );
    }

    embed->audio.resize( lm->num_audio_codebooks );
    for (int cb_index = 0; cb_index < lm->num_audio_codebooks; cb_index++) {
        auto audio_emb = moshi_scaled_embedding_build( ctx,
            lm->emb[cb_index], &embed->audio[cb_index] );

        input = ggml_add( ctx, input, audio_emb );
    }

    if (sum_condition) {
        input = ggml_add( ctx, sum_condition, input );
    }

    return input;
}

void moshi_lmmodel_text_token_embed_step(
        GraphContext & ctx,
        moshi_lmmodel_t * lm,
        lmmodel_embed_t * embed,
        std::vector<int> & sequence
    ) {

    if ( lm->demux_second_stream ) {
        moshi_scaled_embedding_demux_step( ctx, lm->text_emb_demux,
            &embed->text_demux, sequence[0] );
    } else {
        moshi_scaled_embedding_step( ctx, lm->text_emb, &embed->text,
            sequence[0] );
    }

    for (int cb_index = 0; cb_index < lm->num_audio_codebooks; cb_index++) {
        moshi_scaled_embedding_step( ctx,
            lm->emb[cb_index],
            &embed->audio[cb_index],
            sequence[cb_index + lm->audio_offset] );
    }
}

ggml_tensor * moshi_lmmodel_text_token_embed(
        ScratchContext & ctx,
        moshi_lmmodel_t * lm,
        std::vector<int> & sequence,
        ggml_tensor * sum_condition
    ) {
    //ProfileScope profile(time_text_emb_us);

    auto input = lm->demux_second_stream?
        moshi_scaled_embedding_demux( ctx, lm->text_emb_demux, sequence[0] ) :
        moshi_scaled_embedding( ctx, lm->text_emb, sequence[0] );

    for (int cb_index = 0; cb_index < lm->num_audio_codebooks; cb_index++) {
        auto audio_emb = moshi_scaled_embedding( ctx, lm->emb[cb_index],
            sequence[cb_index + lm->audio_offset] );

        input = ggml_add( ctx, input, audio_emb );
    }

    if (sum_condition) {
        input = ggml_add( ctx, sum_condition, input );
    }

    return input;
}

// moshi.models.lm.LMModel.forward_text
std::tuple<ggml_tensor*, ggml_tensor*> moshi_lmmodel_forward_text(
        ScratchContext & ctx,
        moshi_lmmodel_t * lm,
        moshi_lmmodel_states_t * state,
        std::vector<int> & sequence,
        ggml_tensor * sum_condition
    ) {
    //ProfileScope profile(time_forward_text_us);
    //assert len(sequence) == lm.num_codebooks

    auto input = moshi_lmmodel_text_token_embed( ctx, lm, sequence, sum_condition );

    auto transformer_out = moshi_streaming_transformer_graph( ctx,
        lm->transformer, state->transformer, input );

    if ( lm->out_norm )
        transformer_out = moshi_rms_norm( ctx, lm->out_norm, transformer_out );

    auto text_logits = torch_nn_linear( ctx, lm->text_linear, transformer_out );

    return { transformer_out, text_logits };
}

std::tuple<ggml_tensor*, ggml_tensor*> moshi_lmmodel_forward_text_build(
        GraphContext & ctx,
        moshi_lmmodel_t * lm,
        moshi_lmmodel_states_t * state,
        ggml_tensor * sum_condition
    ) {
    auto input = moshi_lmmodel_text_token_embed_build( ctx, lm, &state->embed, sum_condition );

    state->transformer_T = (int)input->ne[1];
    auto transformer_out = moshi_streaming_transformer_graph_build( ctx,
        lm->transformer, state->transformer, input );

    if ( lm->out_norm )
        transformer_out = moshi_rms_norm( ctx, lm->out_norm, transformer_out );

    auto text_logits = torch_nn_linear( ctx, lm->text_linear, transformer_out );

    return { transformer_out, text_logits };
}

void moshi_lmmodel_forward_text_step(
        GraphContext & gctx,
        ScratchContext & ctx,
        moshi_lmmodel_t * lm,
        moshi_lmmodel_states_t * state,
        std::vector<int> & sequence
    ) {
    moshi_lmmodel_text_token_embed_step( gctx, lm, &state->embed, sequence );

    moshi_streaming_transformer_graph_step( ctx,
        lm->transformer, state->transformer, state->transformer_T );
}

// personaplex

std::tuple<ggml_tensor*, ggml_tensor*> moshi_lmmodel_forward_embedding(
        ScratchContext & ctx,
        moshi_lmmodel_t * lm,
        moshi_lmmodel_states_t * state,
        ggml_tensor * input
    ) {
    auto transformer_out = moshi_streaming_transformer_graph( ctx,
        lm->transformer, state->transformer, input );

    if ( lm->out_norm )
        transformer_out = moshi_rms_norm( ctx, lm->out_norm, transformer_out );

    auto text_logits = torch_nn_linear( ctx, lm->text_linear, transformer_out );

    return { transformer_out, text_logits };
}

// moshi.models.lm.LMGen

const int lm_ungenerated_token_id = -2;

struct moshi_lmgen_state_t {
    int offset;
    int skip;
    std::vector<std::vector<int>> cache;
    // Parallel to `cache`: was this slot's value SUPPLIED by the caller rather than
    // sampled by the model? Without it the depformer's own prediction of the user's
    // audio overwrites the user's actual audio in seven of its eight codebooks every
    // frame -- see moshi_lmgen_commit.
    std::vector<std::vector<char>> provided;
    std::vector<int> initial;
};

// Put a generator state into its virgin condition, REUSING whatever storage it
// already has: every resize/assign below is a no-op on a state that has run before,
// so this allocates nothing and frees nothing.
//
// Factored out of moshi_lmgen_state so that the allocator and moshi_lm_reset cannot
// drift apart -- a reset that seeds the cache differently from a fresh state is a
// reset that leaks one conversation into the next, which is the whole hazard.
void moshi_lmgen_state_seed( moshi_lmgen_state_t * state, moshi_lmmodel_t * lm ) {
    state->offset = 0;
    state->skip   = 0;
    int cache_capacity = lm->max_delay + 2;
    if ( lm->personaplex )
        cache_capacity += 1;
    state->cache.resize( cache_capacity );
    state->provided.resize( cache_capacity );
    for (int c = 0; c < cache_capacity; c++) {
        auto & cache = state->cache[c];
        cache.resize( lm->num_codebooks );
        for (int k = 0; k < lm->num_codebooks; k++) {
            cache[k] = lm_ungenerated_token_id;
        }
        state->provided[c].assign( lm->num_codebooks, 0 );
    }
    state->initial.resize( lm->num_codebooks );
    state->initial[0] = lm->text_initial_token_id;
    for ( int i = 1; i < lm->num_codebooks; i++ )
        state->initial[i] = lm->initial_token_id;
    // The reference has a step at offset 0 that seeds SLOT 0 ONLY with the initial
    // tokens and returns nothing (LMGen.prepare_step_input's `offset == 0` branch,
    // `state.cache[:, :, 0] = state.initial[:, :, 0]`). This port has no such step --
    // its offset counter is the reference's minus one -- so that one slot is seeded
    // here instead, and the first real step reads it as its input exactly as the
    // reference's step 1 does. Every OTHER slot stays at lm_ungenerated_token_id,
    // which is deliberate: the delay pipeline fills them, and a slot that is read
    // before anything wrote it is a bug worth crashing on rather than a slot quietly
    // holding a plausible token.
    for ( int k = 0; k < lm->num_codebooks; k++ )
        state->cache[0][k] = state->initial[k];
}

moshi_lmgen_state_t * moshi_lmgen_state( moshi_lmmodel_t * lm ) {
    auto state = new moshi_lmgen_state_t { 0, 0 };
    moshi_lmgen_state_seed( state, lm );
    return state;
}

struct voice_t {
    // Both raw and both OURS: moshi_lm_personaplex_load_voice ggml_init()s the
    // context and ggml_backend_alloc_ctx_tensors() the buffer into it. Every
    // construction site NULLs them first, so the destructor's guards are real.
    ggml_context * ctx;
    ggml_backend_buffer * buffer;
    ggml_tensor * sum;
    ggml_tensor * cross;
    std::deque<int> text_prefixes;
    std::deque<std::vector<int>> audio_prefixes;
    // personaplex
    ggml_tensor * prompt_embeddings;
    ggml_tensor * prompt_cache;
    std::vector<int> text_prompt_tokens;

    ~voice_t() {
        if ( buffer )
            ggml_backend_buffer_free( buffer );
        if ( ctx )
            ggml_free( ctx );
    }
};

// ---------------------------------------------------------------------------
// The cache/delay discipline, in one place.
//
// Every step of this model -- the conversation steps and the system-prompt steps
// alike -- has to agree about four things: where a caller-supplied token lands in
// the delay pipeline, which slot the network reads, which slot the results go to,
// and when the offset advances. The reference implements that once, in
// LMGen.prepare_step_input / process_transformer_output
// (NVIDIA/personaplex moshi/models/lm.py). This port had implemented it three times
// -- once in moshi_lmgen_step and once in each prompt stepper -- and the three did
// not agree. These helpers are the single copy; all three sites call them.
//
// OFFSET CONVENTION. This port's `state->offset` is the reference's offset MINUS
// ONE: where the reference reads cache[P-1], writes results at cache[P] and then
// advances, this port reads cache[O], writes at cache[O+1] and then advances. Every
// index below is the reference's formula with P = O + 1 already substituted, which
// is why the provided-token write is at offset + 1 + delays[k] and not at
// offset + delays[k].
//
// The reference's offset-0 step -- the one that seeds slot 0 and returns None -- has
// no counterpart here, and that is what the minus-one convention IS: this port's
// step O does the work of the reference's step O+1, so the reference's step 0 has
// nothing to do and is folded into moshi_lmgen_state()'s slot-0 seeding. A reader
// comparing step counts between the two will find this port one short, and that is
// the reason.

inline int moshi_lmgen_input_position( moshi_lmgen_state_t * state ) {
    return state->offset % (int) state->cache.size();
}

inline int moshi_lmgen_target_position( moshi_lmgen_state_t * state ) {
    return ( state->offset + 1 ) % (int) state->cache.size();
}

// Supply a token for codebook `k`. It enters the delay pipeline at that codebook's
// own delay, and is FORCED rather than sampled when its step comes round.
inline void moshi_lmgen_provide( moshi_lmgen_state_t * state, moshi_lmmodel_t * lm,
                                 int k, int value ) {
    const int CT = (int) state->cache.size();
    const int position = ( state->offset + 1 + lm->delays[k] ) % CT;
    state->cache[position][k] = value;
    state->provided[position][k] = 1;
}

// Codebooks still inside their own delay window have no generated value to offer
// yet, so the initial token stands in for them (reference: the `offset <= delay`
// loop in prepare_step_input).
inline void moshi_lmgen_provide_initial( moshi_lmgen_state_t * state,
                                         moshi_lmmodel_t * lm ) {
    const int target = moshi_lmgen_target_position( state );
    for ( int k = 0; k < lm->num_codebooks; k++ ) {
        if ( state->offset < lm->delays[k] ) {
            state->cache[target][k] = state->initial[k];
            state->provided[target][k] = 1;
        }
    }
}

inline void moshi_lmgen_read_input( moshi_lmgen_state_t * state, moshi_lmmodel_t * lm,
                                    std::vector<int> & input ) {
    const int position = moshi_lmgen_input_position( state );
    input.resize( lm->num_codebooks );
    for ( int k = 0; k < lm->num_codebooks; k++ ) {
        input[k] = state->cache[position][k];
    }
}

// The text token this step actually conditions on: the supplied one when there is
// one, otherwise what the sampler produced.
inline int moshi_lmgen_next_text_token( moshi_lmgen_state_t * state, int sampled ) {
    const int target = moshi_lmgen_target_position( state );
    return state->provided[target][0] ? state->cache[target][0] : sampled;
}

// Commit a step: sampled values fill the slots nobody supplied, supplied values are
// left alone, the consumed input slot's flags are released, and the offset advances.
//
// "Supplied values are left alone" is the whole point. The depformer predicts all
// dep_q codebooks, including the ones carrying the USER's audio -- that is what
// full-duplex training asks of it -- and at inference the user's real audio has to
// win. Writing the depformer's output unconditionally replaced seven of the eight
// user codebooks with the model's guess at what the user was about to say, every
// frame, and the model then conditioned on its own guess.
inline void moshi_lmgen_commit( moshi_lmgen_state_t * state, moshi_lmmodel_t * lm,
                                int sampled_text,
                                const std::vector<int> & sampled_audio ) {
    const int input_position  = moshi_lmgen_input_position( state );
    const int target = moshi_lmgen_target_position( state );
    if ( ! state->provided[target][0] ) {
        state->cache[target][0] = sampled_text;
    }
    for ( int q = 0; q < (int) sampled_audio.size() && q + 1 < lm->num_codebooks; q++ ) {
        if ( ! state->provided[target][q + 1] ) {
            state->cache[target][q + 1] = sampled_audio[q];
        }
    }
    for ( int k = 0; k < lm->num_codebooks; k++ ) {
        state->provided[input_position][k] = 0;
    }
    state->offset++;
}

// Everything one generator needs to decode a script, and nothing else: no device
// memory, no graph node, no tensor. The decoder reads the text logits it already
// exports, asks the machine which candidates are legal on this step, picks the
// best of them by those logits and reports the choice back -- all on the host,
// inside the step, so the compute graph is byte-identical whether a script is live
// or not.
//
// THE LOCK covers one whole step's use of this slot. The choice and the machine
// update happen in the same critical section on the inference thread, so a caller
// setting or clearing a script from another thread can only land BEFORE or AFTER a
// step, never inside one.
struct moshi_script_slot_t {
    std::mutex mu;

    // The script: token ids the caller produced, and its flags marking the pieces
    // that begin a word (index 0 is a word start by definition).
    std::vector<int>  tokens;
    std::vector<char> word_start;

    moshi_script::Options options;
    moshi_script::Machine machine;

    // Which script the machine is walking. Monotonic per generator and never reset
    // -- not by clearing, not by priming, not by a snapshot restore -- so a stale
    // progress read can never be mistaken for a report about a newer script.
    int epoch = 0;

    // Out-of-set choices. `violations` describes the CURRENT script; the total is
    // monotonic per generator, so a script that ends between two progress reads
    // cannot drop a count. Neither is ever clamped.
    int violations = 0;
    int violations_total = 0;

    // Reusable host buffer for the per-step logits read. Sized once, on first use.
    std::vector<float> logits_host;
};

struct moshi_lmgen_t {
    moshi_lmmodel_t * lm;
    bool use_sampling;
    float temp;
    float temp_text;
    int top_k;
    int top_k_text;

    // these are from the TTSModel callback on_text
    StateMachine * machine;
    State * machine_state;

    // these are from LMGenState
    ggml_tensor * condition_sum;

    // these are from the TTSModel callbacks on_text, on_audio
    std::deque<int> * text_prefixes;
    std::deque<std::vector<int>> * audio_prefixes;

    // personaplex mid-conversation text-token injection (see moshi_lmgen_step).
    // Points at a one-shot slot the caller arms before each step; NULL disables
    // the mechanism entirely, which is what every non-personaplex caller gets.
    //
    // ATOMIC because the two ends are different threads: the caller arms the slot
    // from whatever thread decided to inject, and the step consumes it on the
    // inference thread. A plain int there is a data race, and the arm-then-consume
    // handshake is exactly the shape that loses a token when it tears.
    std::atomic<int> * forced_text_token;

    // Scripted (constrained) text decoding. Points at the slot the generator owns;
    // NULL disables the mechanism entirely, which is what every non-personaplex
    // caller gets and what keeps the state-machine (TTS) path untouched. A slot
    // whose machine holds no live script costs nothing either: the step's branch is
    // skipped and the graph's own sampled token stands.
    moshi_script_slot_t * script;
};

bool moshi_lmgen_step(
        ScratchContext & scratch,
        moshi_lmgen_t * lmgen,
        moshi_lmgen_state_t * state,
        moshi_lmmodel_states_t * lm_states,
        bool depformer_replace_tokens,
        int & int_text_token,
        std::vector<int> & int_audio_tokens,
        float * vad = NULL,
        int skip_prefix = 2 // for debugging set to 0
) {
    auto lm = lmgen->lm;
    auto use_sampling = lmgen->use_sampling;
    auto temp = lmgen->temp;
    auto temp_text = lmgen->temp_text;
    auto top_k = lmgen->top_k;
    auto top_k_text = lmgen->top_k_text;
    auto machine = lmgen->machine;
    auto machine_state = lmgen->machine_state;
    auto condition_sum = lmgen->condition_sum;
    auto text_prefixes = lmgen->text_prefixes;
    auto audio_prefixes = lmgen->audio_prefixes;
    auto forced_text_token = lmgen->forced_text_token;
    //ProfileScope profile(time_lmgen_step_us);
    const int CT = (int) state->cache.size();
    int dep_q = lm->dep_q;
    if ( lm->personaplex )
        dep_q = 8;
    int dep_q_1 = dep_q + 1;

    auto needed_tokens = lm->num_codebooks - dep_q - 1;
    if ( needed_tokens > 0 ) {
        assert( (int)int_audio_tokens.size() >= needed_tokens );
        assert( (int)lm->delays.size() >= needed_tokens );
        int start = dep_q_1;
        for ( int i = 0; i < needed_tokens; i++ ) {
            // Through the shared discipline, which puts this one slot later than the
            // hand-rolled write it replaces -- see the offset convention note there.
            moshi_lmgen_provide( state, lm, start + i, int_audio_tokens[i] );
        }
    }

    // Arm an injected text token the same way the reference does -- as a PROVIDED
    // token in the delay pipeline (LMGen.step's `text_token=` argument), not as a
    // post-sampling patch. delays[0] is 0, so it lands on this step's target slot and
    // takes effect immediately, exactly as before; routing it here means the forcing
    // rule lives in one place and the injected token reaches the cache the same way
    // every other supplied token does.
    if ( forced_text_token ) {
        const int forced = forced_text_token->exchange( -1, std::memory_order_acq_rel );
        if ( forced >= 0 ) {
            moshi_lmgen_provide( state, lm, 0, forced );
        }
    }

    // Initial-token fill LAST, after every caller-supplied token, mirroring
    // prepare_step_input's own order. On this model the order cannot matter -- a
    // codebook inside its delay window and a codebook the caller supplies are
    // disjoint sets here, because delays[0] is 0 so the text stream is never inside
    // one -- but that is an accident of these delays, not a property of the
    // algorithm, and matching the reference costs nothing.
    moshi_lmgen_provide_initial( state, lm );
    /*
    it would possibly make sense to "warm up" the cache to avoid the branching
    logic below, and maybe have a more complete graph. may not be a performance
    benefit to it though.
    UPDATE: after investigating, there is additional logic later on that looks
    for -1 and zeroes out the results if present. that means to do that you
    need to modify data. see: scaled_embedding functions
    */
    std::vector<int> input;
    moshi_lmgen_read_input( state, lm, input );

    ONCE( scratch.set_name("text") );
    ON_NTH( 32, scratch.set_name( "text_32" ) );

#ifdef USE_SCRATCH
    auto [scratch_transformer_out, text_logits] = moshi_lmmodel_forward_text(
        scratch, lm, lm_states,
        input, condition_sum );

    auto cpy_transformer_out = ggml_cpy( scratch,
        scratch_transformer_out, lm_states->transformer_out );
    scratch.build_forward_expand( cpy_transformer_out );

    // note this does the compute
    auto text_token = moshi_sample_token_int( scratch, text_logits,
        use_sampling, temp_text, top_k_text );
#else
    if ( ! lm_states->gctx ) {
        lm_states->gctx = new GraphContext( 256, scratch.backend );
        GraphContext &graph = *lm_states->gctx;

        auto [graph_transformer_out, text_logits] = moshi_lmmodel_forward_text_build(
            graph, lm, lm_states, condition_sum );

        auto cpy_transformer_out = ggml_cpy( graph,
            graph_transformer_out, lm_states->transformer_out );
        graph.build_forward_expand( cpy_transformer_out );

        lm_states->sampler_out = moshi_sample_token( graph, text_logits,
            use_sampling, temp_text, top_k_text );

        // Keep the logits as a graph output too. They are already computed -- the
        // sampler reads them -- so this materialises an existing value rather than
        // adding arithmetic, and the token streams are bit-identical with and
        // without it (verified: the dumps used to derive the M3 parity margins
        // produced the same 64 ids as the runs without them).
        lm_states->text_logits_out = text_logits;
        graph.build_forward_expand( text_logits );

        graph.build_forward_expand( lm_states->sampler_out );
        graph.alloc();

        lm_states->conv_graph = lm_states->transformer->graph;
    }

    GraphContext &graph = *lm_states->gctx;
    // Restore the conversation view of the shared slot: the prompt path uses the
    // same field for its own, differently-owned graph.
    lm_states->transformer->graph = lm_states->conv_graph;
    moshi_lmmodel_forward_text_step( graph, scratch, lm, lm_states, input );

    scratch.compute();
    graph.compute();

    int text_token;
    ggml_backend_tensor_get( lm_states->sampler_out, &text_token, 0, 4 );
#endif

    // The SAMPLED token, kept separate from the one this step conditions on: the
    // cache records what the model produced, the depformer consumes what was
    // supplied when something was (reference: `sampled_text_token` vs
    // `next_text_token` in process_transformer_output).
    const int sampled_text_token = text_token;
    text_token = moshi_lmgen_next_text_token( state, sampled_text_token );

    // on_text_hook
    if ( machine ) {
        if ( text_prefixes && text_prefixes->size() ) {
            text_token = text_prefixes->front();
            text_prefixes->pop_front();
        } else {
            static int prev_in_token = 3;
            static int prev_text_token = 3;
            auto in_token = text_token;

            text_token = machine->process(state->offset, machine_state, text_token);

            if (machine->logging && ( prev_in_token != in_token || prev_text_token != text_token ) ) {
                printf( "%d {%d, %d}\n", in_token, text_token % 8001, text_token / 8001 - 1 );
                prev_in_token = in_token;
                prev_text_token = text_token;
            }
        }
    }

    int_audio_tokens.resize( lm->dep_q );
    if ( lm->depformer ) {
        if (!depformer_replace_tokens) {
            //ProfileScope profile(time_depformer_step_us);
            moshi_lmmodel_depformer_step(
                scratch, lm, lm_states,
                text_token, use_sampling, temp, top_k,
                int_audio_tokens );
        } else {
            for (int i = 0; i < (int)int_audio_tokens.size(); i++) {
                int_audio_tokens[i] = -1;
            }
        }
        // on_audio_hook
        const int delay_steps = lm->delay_steps;
        if ( delay_steps ) {
            for (int q = 0; q < (int)int_audio_tokens.size(); q++) {
                if (state->offset < lm->delays[q + 1] + delay_steps)
                    int_audio_tokens[q] = -1; // token_ids.zero
            }
        }
        if ( audio_prefixes && audio_prefixes->size() ) {
            state->skip = skip_prefix;
            auto audio_codes = audio_prefixes->front();
            for (int q = 0; q < (int)int_audio_tokens.size(); q++) {
                if (audio_codes[q] != lm_ungenerated_token_id)
                    int_audio_tokens[q] = audio_codes[q];
            }
            audio_prefixes->pop_front();
        }
    }

    // WHAT THE CACHE MUST RECORD is the text token this step actually conditioned the
    // depformer on -- the monologue as it was spoken -- because the cache IS the
    // model's account of its own turn: every later step attends to it, and
    // moshi_lm_receive reads the emitted text token straight back out of it.
    //
    // Without a state machine `text_token` IS `sampled_text_token` (the only other
    // writer is moshi_lmgen_next_text_token, and a SUPPLIED token is already in the
    // slot -- moshi_lmgen_commit leaves any slot whose `provided` flag is set exactly
    // as the caller wrote it). So for every PersonaPlex generator this is identical to
    // what it replaced, token for token: moshi_lm_start only builds a StateMachine on
    // the non-personaplex voice path.
    //
    // WITH a state machine the two diverge, and that is the bug this replaces. The
    // machine rewrites the token -- padding discipline, word starts, and the
    // text_prefixes it splices for a scripted voice -- and the depformer voices the
    // REWRITE, one line above. Committing the sampler's token instead left the model
    // conditioned on a monologue it never spoke and could not see, and the tap read
    // that phantom stream back out. The reference mutates its `text_token` tensor
    // in place inside the on_text hook and then caches THAT tensor, which is this.
    const int committed_text_token = machine ? text_token : sampled_text_token;
    moshi_lmgen_commit( state, lm, committed_text_token,
        lm->depformer ? int_audio_tokens : std::vector<int>() );

    
    if ( state->skip > 0 ) {
        --state->skip;
        return false;
    }

    if (state->offset <= lm->max_delay || depformer_replace_tokens)
        return false;

    //int_audio_tokens[0] = state->cache[(state->offset - lm->max_delay) % state->cache.size()][1];
    int index = (state->offset - lm->max_delay + lm->delays[0]) % CT;
    int_text_token = state->cache[index][0];
    for ( int i = 1; i < dep_q_1; i++ ) {
        index = (state->offset - lm->max_delay + lm->delays[i]) % CT;
        int_audio_tokens[i - 1] = state->cache[index][i];
    }

    for (auto x : int_audio_tokens) {
        if (x == -1)
            return false;
    }
    
    if ( vad ) {
        if ( lm->extra_heads.size() > 2 ) {
            auto linear = torch_nn_linear( scratch, lm->extra_heads[2], lm_states->transformer_out );
            auto soft_max = ggml_soft_max( scratch, linear );
            auto view = ggml_view_1d( scratch, soft_max, 1, 0 );
            scratch.build_forward_expand( view, vad );
            scratch.compute();
        } else {
            *vad = 0;
        }
    }

    return true;
}

// personaplex system prompts

std::vector<int> SILENCE_TOKENS = { 948, 243, 1178, 546, 1736, 1030, 1978, 2008 };
std::vector<int> SINE_TOKENS    = { 430, 1268, 381, 1611, 1095, 1495, 56, 472 };

// Number of codebooks per stream in a personaplex frame: [audio_offset ..
// audio_offset+8) is the MOSHI stream and the 8 above it are the USER stream.
// Both SILENCE_TOKENS and SINE_TOKENS are exactly that wide.
const int PERSONAPLEX_TOKENS_PER_STREAM = 8;

// Supply a system-prompt step's audio: silence on the moshi stream, sine on the
// user stream, both through the shared delay pipeline.
//
// SINE_TOKENS IS APPLIED HERE NOW, and the history is worth keeping because it is a
// good example of a fix that looks obvious and is not. The reference passes
// moshi_tokens=SILENCE_TOKENS *and* input_tokens=SINE_TOKENS to every system-prompt
// step (LMGen._step_audio_silence_core, _step_text_prompt_core), while this port
// declared SINE_TOKENS and never read it. Applying it while the steppers still had
// their own broken cache discipline made the model INTERRUPT the user 320 ms in with
// "Hey, let me know" and then fall silent -- worse than leaving it out, which is why
// an earlier revision of this series deliberately left it out and said so. With the
// steppers going through moshi_lmgen_provide() the phase error is gone and the sine
// is simply what the reference does.
void moshi_lmgen_provide_prompt_audio( moshi_lmgen_state_t * state,
                                       moshi_lmmodel_t * lm ) {
    const int per_stream = PERSONAPLEX_TOKENS_PER_STREAM;
    for ( int i = 0; i < per_stream && i < lm->num_audio_codebooks; i++ ) {
        moshi_lmgen_provide( state, lm, i + lm->audio_offset, SILENCE_TOKENS[i] );
    }
    for ( int i = 0; i < per_stream && per_stream + i < lm->num_audio_codebooks; i++ ) {
        moshi_lmgen_provide( state, lm, i + lm->audio_offset + per_stream,
            SINE_TOKENS[i] );
    }
}

void moshi_lmgen_step_voice_prompt(
    ScratchContext & scratch,
    moshi_lmgen_t * lmgen,
    moshi_lmgen_state_t * state,
    moshi_lmmodel_states_t * lm_states,
    voice_t * voice
) {
    if ( ! voice )
        return;
    if ( voice->prompt_embeddings ) {
        auto use_sampling = lmgen->use_sampling;
        auto temp = lmgen->temp;
        auto temp_text = lmgen->temp_text;
        auto top_k = lmgen->top_k;
        auto top_k_text = lmgen->top_k_text;
        std::vector<int> int_audio_tokens( lmgen->lm->dep_q );
        auto lm = lmgen->lm;
        for ( int i = 0; i < voice->prompt_embeddings->ne[3]; i++ ) {
            // The reference replays a stored voice prompt through the SAME
            // bookkeeping every other step uses: LMGen.step_embeddings() calls
            // prepare_step_input() with the initial token on every audio codebook
            // and zero_text_code on the text stream, uses only its provided/target
            // outputs, and then runs the transformer on the embedding instead of on
            // the token input. That is what this reproduces.
            //
            // Skipping it -- which is what a bare offset++ here amounts to -- does
            // not just lose bookkeeping for these frames. It leaves the `provided`
            // mask EMPTY going into the first silence step, where the reference has
            // it set for the 14 delay-1 codebooks; the reference forces those slots
            // out of the restored voice cache and this port sampled them instead.
            moshi_lmgen_provide( state, lm, 0, lm->text_padding_token_id );
            for ( int k = 1; k < lm->num_codebooks; k++ ) {
                // _get_initial_token() for every audio codebook. The reference
                // splits it across input_tokens/moshi_tokens, which lands the two
                // halves on each other's streams; that is invisible because
                // state->initial is the same value for every audio codebook.
                moshi_lmgen_provide( state, lm, k, state->initial[k] );
            }
            moshi_lmgen_provide_initial( state, lm );

            auto input = ggml_view_4d( scratch, voice->prompt_embeddings,
                voice->prompt_embeddings->ne[0],
                voice->prompt_embeddings->ne[1],
                voice->prompt_embeddings->ne[2],
                1,
                voice->prompt_embeddings->nb[1],
                voice->prompt_embeddings->nb[2],
                voice->prompt_embeddings->nb[3],
                voice->prompt_embeddings->nb[3] * i
            );
            input = ggml_cast( scratch, input, GGML_TYPE_F32 );
            auto [scratch_transformer_out, text_logits] = moshi_lmmodel_forward_embedding(
                scratch, lmgen->lm, lm_states, input );

            auto cpy_transformer_out = ggml_cpy( scratch,
                scratch_transformer_out, lm_states->transformer_out );
            scratch.build_forward_expand( cpy_transformer_out );

            // This call also DRIVES THE COMPUTE for the transformer copy above; it
            // is not a wasted sample, and deleting it breaks the step.
            const int sampled_text_token = moshi_sample_token_int( scratch, text_logits,
                use_sampling, temp_text, top_k_text );
            const int text_token = moshi_lmgen_next_text_token( state, sampled_text_token );

            moshi_lmmodel_depformer_step(
                scratch, lm, lm_states,
                text_token, use_sampling, temp, top_k,
                int_audio_tokens );

            moshi_lmgen_commit( state, lm, sampled_text_token, int_audio_tokens );
        }

        // The stored cache is restored over the VALUES only, exactly as the
        // reference's `state.cache.copy_(self.voice_prompt_cache)` does -- it touches
        // cache and not provided. The flags the last replayed step left behind are
        // therefore live, and they are what makes the first silence step force its
        // delay-1 codebooks out of this restored cache instead of sampling them.
        //
        // note that dimensions inverted
        assert( voice->prompt_cache->type == GGML_TYPE_I32 );
        int cache_height = (int)state->cache.size(); // time
        int cache_width = (int)state->cache[0].size();
        assert( cache_height == voice->prompt_cache->ne[0] );
        assert( cache_width == voice->prompt_cache->ne[1] );
        assert( voice->prompt_cache->ne[2] == 1 );
        assert( voice->prompt_cache->ne[3] == 1 );
        std::vector<int32_t> cache( ggml_nelements( voice->prompt_cache ) );
        ggml_backend_tensor_get( voice->prompt_cache, cache.data(), 0, ggml_nbytes( voice->prompt_cache) );
        for ( int i = 0; i < cache_height; i++ ) {
            for ( int j = 0; j < cache_width; j++ ) {
                state->cache[i][j] = cache[ i + j * cache_height ];
            }
        }
    }
}

// Teacher-force the tokenized system text prompt.
void moshi_lmgen_step_text_prompt_tokens(
    ScratchContext & scratch,
    moshi_lmgen_t * lmgen,
    moshi_lmgen_state_t * state,
    moshi_lmmodel_states_t * lm_states,
    std::vector<int> & tokens
) {
    if ( tokens.empty() )
        return;

    auto lm = lmgen->lm;
    auto use_sampling = lmgen->use_sampling;
    auto temp = lmgen->temp;
    auto temp_text = lmgen->temp_text;
    auto top_k = lmgen->top_k;
    auto top_k_text = lmgen->top_k_text;
    std::vector<int> int_audio_tokens( lm->dep_q );

    for ( int t = 0; t < (int)tokens.size(); t++ ) {
        // Supply this step: the prompt token on the text stream, silence on the
        // moshi stream, sine on the user stream -- all through the shared pipeline,
        // so the network reads them at the same slots moshi_lmgen_step would.
        moshi_lmgen_provide( state, lm, 0, tokens[t] );
        moshi_lmgen_provide_prompt_audio( state, lm );
        moshi_lmgen_provide_initial( state, lm );

        std::vector<int> input;
        moshi_lmgen_read_input( state, lm, input );

        auto [scratch_transformer_out, text_logits] = moshi_lmmodel_forward_text(
            scratch, lm, lm_states, input, NULL );

        auto cpy_transformer_out = ggml_cpy( scratch,
            scratch_transformer_out, lm_states->transformer_out );
        scratch.build_forward_expand( cpy_transformer_out );

        // The sample is kept -- it is what the cache records for an unsupplied slot
        // -- while the depformer conditions on the supplied prompt token. Note this
        // call also DRIVES THE COMPUTE for the transformer copy above; it is not a
        // wasted sample, and deleting it breaks the step.
        const int sampled_text_token = moshi_sample_token_int( scratch, text_logits,
            use_sampling, temp_text, top_k_text );
        const int text_token = moshi_lmgen_next_text_token( state, sampled_text_token );

        moshi_lmmodel_depformer_step(
            scratch, lm, lm_states,
            text_token, use_sampling, temp, top_k,
            int_audio_tokens );

        moshi_lmgen_commit( state, lm, sampled_text_token, int_audio_tokens );
    }
}

void moshi_lmgen_step_audio_silence(
    ScratchContext & scratch,
    moshi_lmgen_t * lmgen,
    moshi_lmgen_state_t * state,
    moshi_lmmodel_states_t * lm_states,
    int n_steps
) {
    auto lm = lmgen->lm;
    auto use_sampling = lmgen->use_sampling;
    auto temp = lmgen->temp;
    auto temp_text = lmgen->temp_text;
    auto top_k = lmgen->top_k;
    auto top_k_text = lmgen->top_k_text;
    std::vector<int> int_audio_tokens( lm->dep_q );

    for ( int i = 0; i < n_steps; i++ ) {
        moshi_lmgen_provide( state, lm, 0, lm->text_padding_token_id );
        moshi_lmgen_provide_prompt_audio( state, lm );
        moshi_lmgen_provide_initial( state, lm );

        std::vector<int> input;
        moshi_lmgen_read_input( state, lm, input );

        auto [scratch_transformer_out, text_logits] = moshi_lmmodel_forward_text(
            scratch, lm, lm_states, input, NULL );

        auto cpy_transformer_out = ggml_cpy( scratch,
            scratch_transformer_out, lm_states->transformer_out );
        scratch.build_forward_expand( cpy_transformer_out );

        const int sampled_text_token = moshi_sample_token_int( scratch, text_logits,
            use_sampling, temp_text, top_k_text );
        const int text_token = moshi_lmgen_next_text_token( state, sampled_text_token );

        moshi_lmmodel_depformer_step(
            scratch, lm, lm_states,
            text_token, use_sampling, temp, top_k,
            int_audio_tokens );

        moshi_lmgen_commit( state, lm, sampled_text_token, int_audio_tokens );
    }
}

// Reference default for the two silence slots that bracket the text prompt.
//
// This was 8 ("~0.64s at 12.5fps"). The reference implementation's LMGen defaults
// audio_silence_frame_cnt to 1 and its server exposes it as a per-session knob, so
// 1 is the value a prefix has to use to agree with the reference token for token.
// moshi_lm_start() takes it as a parameter, so the old 8 is still reachable.
const int PERSONAPLEX_AUDIO_SILENCE_FRAMES = 1;

void moshi_lmgen_step_system_prompts(
    ScratchContext & ctx,
    moshi_lmgen_t * lmgen,
    moshi_lmgen_state_t * state,
    moshi_lmmodel_states_t * lm_states,
    voice_t * voice,
    int audio_silence_frames = PERSONAPLEX_AUDIO_SILENCE_FRAMES
) {
    moshi_lmgen_step_voice_prompt( ctx, lmgen, state, lm_states, voice );
    moshi_lmgen_step_audio_silence( ctx, lmgen, state, lm_states, audio_silence_frames );
    if ( voice && voice->text_prompt_tokens.size() > 0 ) {
        moshi_lmgen_step_text_prompt_tokens( ctx, lmgen, state, lm_states,
            voice->text_prompt_tokens );
    }
    moshi_lmgen_step_audio_silence( ctx, lmgen, state, lm_states, audio_silence_frames );
}
