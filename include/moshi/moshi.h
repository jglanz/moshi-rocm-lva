#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <deque>

#include <sentencepiece_processor.h>

#include "ptrs.h"
#include "safetensor.h"

#if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef MOSHI_BUILD
#        define MOSHI_API __declspec(dllexport) extern
#    else
#        define MOSHI_API __declspec(dllimport) extern
#    endif
#else
#    define MOSHI_API __attribute__ ((visibility ("default"))) extern
#endif

// MARK: Moshi Context

struct moshi_context_t;

MOSHI_API moshi_context_t * moshi_alloc( ggml_backend * backend, ggml_backend * backend_cpu );
MOSHI_API void unref( moshi_context_t * moshi );

// MARK: Mimi Codec

struct mimi_codec_t;

MOSHI_API mimi_codec_t * mimi_alloc( moshi_context_t * moshi, const char * filename, int n_q );
MOSHI_API void unref( mimi_codec_t * codec );
MOSHI_API float mimi_frame_rate( mimi_codec_t * codec );
MOSHI_API int mimi_frame_size( mimi_codec_t * codec );
MOSHI_API void mimi_save_gguf( mimi_codec_t * codec, const char * filepath );

// MARK: Mimi Encode

struct mimi_encode_context_t;

MOSHI_API mimi_encode_context_t * mimi_encode_alloc_context( mimi_codec_t * codec );
MOSHI_API void unref( mimi_encode_context_t * context );
MOSHI_API void mimi_encode_reset( mimi_encode_context_t * context );
MOSHI_API void mimi_encode_send( mimi_encode_context_t * context, float * frame );
MOSHI_API void mimi_encode_receive( mimi_encode_context_t * context, int16_t * tokens );

// MARK: Mimi Decode

struct mimi_decode_context_t;

MOSHI_API mimi_decode_context_t * mimi_decode_alloc_context( mimi_codec_t * codec );
MOSHI_API void unref( mimi_decode_context_t * context );
MOSHI_API void mimi_decode_reset( mimi_decode_context_t * context );
MOSHI_API void mimi_decode_send( mimi_decode_context_t * context, int16_t * tokens );
MOSHI_API void mimi_decode_receive( mimi_decode_context_t * context, float * frame );

// MARK: Tokenizer

struct Entry {
    std::vector<int> tokens;
    std::string text;
    int padding;
    int64_t time = 0;
};

struct tokenizer_t;

MOSHI_API tokenizer_t * tokenizer_alloc( const char * filepath, bool insert_bos = true );
MOSHI_API void unref( tokenizer_t * tok );
MOSHI_API bool tokenizer_empty( tokenizer_t * tok );
MOSHI_API int tokenizer_send( tokenizer_t * tok, std::string text );
MOSHI_API int tokenizer_receive( tokenizer_t * tok, Entry * entry );
MOSHI_API std::string tokenizer_id_to_piece( tokenizer_t * tok, int token );
// Encode a whole string in one SentencePiece call and return the token count.
// tokenizer_send/tokenizer_receive is the streaming word-at-a-time path used to
// drive the TTS state machine; it is not a way to tokenize a string, and there was
// no other way to do it through the public API.
MOSHI_API int tokenizer_encode( tokenizer_t * tok, const char * text, std::vector<int> & tokens );
// tokenizer_id_to_piece() with the SentencePiece word-initial marker U+2581
// turned back into a space -- the monologue tap every tool in tools/ open-codes.
MOSHI_API std::string tokenizer_id_to_text( tokenizer_t * tok, int token );

// MARK: Config

struct config_fuser_t {
    bool cross_attention_pos_emb; // true
    float cross_attention_pos_emb_scale; // 1
    std::vector<std::string> sum; // [ "control", "cfg" ]
    // ? "prepend": [],
    std::vector<std::string> cross; // [ "speaker_wavs" ]
};

struct config_tts_t {
    float audio_delay; // 1.28
    int64_t second_stream_ahead; // 2
};

struct config_stt_t {
    float audio_delay_seconds; // 0.5
    float audio_silence_prefix_seconds; // 0.0
};

struct config_model_id_t {
    std::string sig;
    int64_t epoch;
};

struct config_lm_gen_t {
    float temp; // 0.6
    float temp_text; // 0.6
    int64_t top_k; // 250
    int64_t top_k_text; // 50
};

struct moshi_config_t {
    int64_t card; // 2048
    int64_t n_q; // 32 || 16
    int64_t dep_q; // 32 || 16
    std::vector<int64_t> delays; // 32 || 16
    int64_t dim; // 2048 || 1024
    int64_t text_card; // 8000
    int64_t existing_text_padding_id; // 3
    int64_t num_heads; // 16
    int64_t num_layers; // 16 || 24
    float hidden_scale; // 4.125
    bool causal; // true
    // layer_scale = NULL;
    int64_t context; // 500
    int64_t max_period; // 10000
    std::string gating;// "silu"
    std::string norm; // "rms_norm_f32"
    std::string positional_embedding; // "rope"
    int64_t depformer_dim; // 1024
    int64_t depformer_num_heads; // 16
    int64_t depformer_num_layers; // 4
    //int64_t depformer_dim_feedforward; // 3072   no needed, it's in the weight files
    bool depformer_multi_linear; // true
    int64_t depformer_context; // 0
    int64_t depformer_max_period; // 0
    std::string depformer_gating; // ""
    std::string depformer_pos_emb; // "none"
    bool depformer_weights_per_step; // true
    int64_t depformer_low_rank_embeddings; // 128
    bool demux_second_stream; // true
    // text_card_out = null || 5
    // conditioners_t * conditioners; // NULL
    config_fuser_t fuser;
    bool cross_attention; // true || false
    int64_t extra_heads_num_heads;
    //int extra_heads_dim; // this will come from weights
    config_tts_t tts_config;
    config_stt_t stt_config;
    config_model_id_t model_id;
    std::vector<int64_t> depformer_weights_per_step_schedule; // 32 || 16
    std::string model_type;
    config_lm_gen_t lm_gen_config;
    std::string tokenizer_name; // "tokenizer_spm_8k_en_fr_audio.model"
    std::string mimi_name; // "tokenizer-e351c8d8-checkpoint125.safetensors",
    std::string moshi_name; // "dsm_tts_1e68beda@240.safetensors" || "dsm_tts_d6ef30c7@1000.safetensors"
};

MOSHI_API int moshi_get_config( moshi_config_t * config, const char * filename );

// MARK: LM

struct moshi_lm_t;

MOSHI_API moshi_lm_t * moshi_lm_from_files(
    moshi_context_t * moshi,
    moshi_config_t * config,
    const char * filepath
);
MOSHI_API void unref( moshi_lm_t * lm );
MOSHI_API void moshi_lm_set_delay_steps( moshi_lm_t * lm, int delay_steps );
MOSHI_API int moshi_lm_get_max_delay( moshi_lm_t * lm );
MOSHI_API int moshi_lm_get_delay_steps( moshi_lm_t * lm );
// Special text-token ids, so a consumer of the monologue tap never has to spell
// them. `text_padding_token_id` comes from the model config's
// `existing_text_padding_id`; the pair {new_word, pad} is what upstream's tools
// filter out of the monologue before printing it.
MOSHI_API int moshi_lm_get_text_card( moshi_lm_t * lm );
MOSHI_API int moshi_lm_get_text_padding_token_id( moshi_lm_t * lm );
MOSHI_API int moshi_lm_get_text_new_word_token_id( moshi_lm_t * lm );
// delays[0]. A token forced at sampling time reaches the moshi_lm_receive
// out-parameter max_delay - text_delay frames later; a tap that needs to align
// forced tokens with observed ones needs both numbers.
MOSHI_API int moshi_lm_get_text_delay( moshi_lm_t * lm );
MOSHI_API bool moshi_lm_quantize( moshi_lm_t * lm, const char * quant );
MOSHI_API int moshi_lm_load( moshi_lm_t * lm );
MOSHI_API void moshi_lm_save_gguf( moshi_lm_t * lm, const char * filepath );

// MARK: Generator

struct moshi_lm_gen_t;

MOSHI_API moshi_lm_gen_t * moshi_lm_generator( moshi_lm_t * lm );
MOSHI_API void unref( moshi_lm_gen_t * gen );

MOSHI_API int moshi_lm_set_voice_condition( moshi_context_t * moshi, moshi_lm_gen_t * gen, const char * filepath );
MOSHI_API int moshi_lm_load_voice_condition( moshi_context_t * moshi, moshi_lm_gen_t * gen );
MOSHI_API int moshi_lm_voice_prefix( moshi_lm_gen_t * gen, std::deque<int> & text_prefix, std::deque<std::vector<int>> & audio_prefix );
MOSHI_API int moshi_lm_personaplex_load_voice( moshi_context_t * moshi, moshi_lm_gen_t * gen, const char * filename );
MOSHI_API void moshi_lm_personaplex_set_text_prompt( moshi_lm_gen_t * gen, tokenizer_t * tok, const char * text );
// Set the system text prefix from token ids the CALLER produced. The reference
// PersonaPlex server wraps the persona script in <system> tags before encoding it;
// that wrapping is application policy, not a model property, so it lives with the
// caller and this entry point takes the finished ids.
MOSHI_API void moshi_lm_personaplex_set_text_prompt_tokens( moshi_lm_gen_t * gen, const int * tokens, int n_tokens );
// Read back whatever prefix is armed. Exists so a parity harness can prove its
// prefix matches the oracle's before blaming the decode loop for a divergence.
MOSHI_API int moshi_lm_personaplex_get_text_prompt_tokens( moshi_lm_gen_t * gen, std::vector<int> & tokens );

// `audio_silence_frames` is the length of each of the two silence slots that
// bracket the personaplex text prompt. 1 is the reference implementation's
// default (LMGen.audio_silence_frame_cnt); the prefix only agrees with the
// reference token for token at that value.
//
// `prime` false ALLOCATES ONLY: the arena, the state context and the generator
// state exist, and nothing has been seeded or primed -- the state tensors still
// hold whatever the allocator handed back. The generator that comes back is NOT
// usable, and the only sanctioned next step is moshi_lm_snapshot_restore() with a
// snapshot taken from a generator on this same model primed with the prompt this
// one wants; that call writes every tensor and every scalar a seed would have.
// (If the restore is refused, moshi_lm_reset() makes the generator usable again.)
// Seeding first would be a full host-to-device pass whose every byte the restore
// overwrites -- 1.5 GiB and 1.8 s on PersonaPlex-7B, which is most of what the
// snapshot exists to save. See the snapshot section below.
MOSHI_API void moshi_lm_start( moshi_context_t * moshi, moshi_lm_gen_t * gen, float depth_temperature, float text_temperature, bool logging = false, int audio_silence_frames = 1, bool prime = true );

// Put a started generator back to a virgin conversation WITHOUT reallocating any of
// its state: the delay cache, the state tensors, the streaming states and the
// injection slot are re-seeded in place and the system-prompt phase runs again with
// whatever voice / text prompt is currently set. Allocates nothing, frees nothing.
//
// This is what a resident server calls between connections instead of destroying
// the generator -- see the comment on moshi_lm_prime in moshi.cpp for why reuse
// rather than free is the design.
// The default MATCHES moshi_lm_start's deliberately: a reset primes the generator
// through the same prompt phase, so a different silence bracket here would mean a
// reset generator was conditioned differently from a fresh one -- silently, and
// only on the second conversation.
// A caller that means to RESTORE a snapshot instead of re-priming does not call
// this at all: moshi_lm_snapshot_restore() writes a superset of what a reset does,
// so a reset before it is pure cost. Call this to prime, or to recover a generator
// whose restore was refused.
MOSHI_API void moshi_lm_reset( moshi_context_t * moshi, moshi_lm_gen_t * gen, int audio_silence_frames = 1 );

// MARK: Primed-state snapshot
//
// THE PROBLEM. The system-prompt phase moshi_lm_prime runs is ONE FULL FORWARD PASS
// PER PROMPT TOKEN, plus the voice-prompt replay and two silence brackets
// (moshi_lmgen_step_system_prompts). Its cost is linear in the persona's length:
// measured on a gfx1100, 25 prompt tokens reach ready in 6.0 s and 282 in 16.3 s,
// and a host that sends its agent instructions as the persona sends a thousand.
// Every reconnect re-pays it, and it recomputes the same thing every time: for a
// FIXED (voice, prompt, silence-bracket) the primed state is a pure function of
// those inputs, sampled greedily at the temperature the config carries.
//
// SO SAVE IT. A snapshot is everything a conversation owns and nothing it shares:
// the state context's tensors (KV caches and streaming buffers, bounded by the
// state context -- NOT by the weights), the transformer's live offset, and the
// generator's delay cache with its `provided` masks. Restoring it puts a generator
// into the state the prompt phase would have left it in, with no forward passes at
// all. moshi_lm_snapshot_bytes() reports what one costs; the blob is host memory,
// so holding a few does not compete with the model for device memory.
//
// SCOPE. The state-machine (TTS) path is NOT covered -- StateMachine's own state is
// not in the snapshot -- so capture and restore both refuse a generator that has
// one, with -2. A snapshot is only valid for the generator's own moshi_lm_t and for
// a state context with the identical registration list; restore checks both and
// refuses with -3 rather than writing a mismatched blob.
//
// CORRECTNESS IS THE CALLER'S HALF. Nothing here knows what prompt produced the
// state it holds. A caller that restores a snapshot taken under a DIFFERENT voice
// or prompt gets a perfectly consistent generator conditioned on the wrong persona,
// with no diagnostic. Key the cache on every input that feeds the prompt phase.
struct moshi_lm_snapshot_t;

MOSHI_API moshi_lm_snapshot_t * moshi_lm_snapshot_alloc();
MOSHI_API void unref( moshi_lm_snapshot_t * snapshot );
// Host bytes this snapshot holds; 0 before a successful capture.
MOSHI_API size_t moshi_lm_snapshot_bytes( moshi_lm_snapshot_t * snapshot );
// What ONE snapshot of this generator would cost, without taking it.
MOSHI_API size_t moshi_lm_snapshot_state_bytes( moshi_lm_gen_t * gen );
// 0 on success; -1 unusable generator/snapshot, -2 state-machine generator.
MOSHI_API int moshi_lm_snapshot_capture( moshi_lm_gen_t * gen, moshi_lm_snapshot_t * snapshot );
// 0 on success; -1 unusable, -2 state-machine generator, -3 the snapshot does not
// belong to this generator's model/state layout.
MOSHI_API int moshi_lm_snapshot_restore( moshi_lm_gen_t * gen, moshi_lm_snapshot_t * snapshot );
// Personaplex mid-conversation text-token injection. Arm the slot with the token
// this frame must emit, then call moshi_lm_receive/moshi_lm_receive2: the step
// overrides the sampled text token with it BEFORE the depformer runs, so the audio
// heads condition on the forced token and the model speaks it in its own voice.
// This is the same mechanism the reference server uses (LMGen.step(text_token=...)).
//
// One shot: every step consumes the slot with an atomic exchange, including steps
// that return 0. `token` must be >= 0; a negative value is the same as clearing.
// The slot is safe to arm from a thread other than the one running the step, which
// is the point -- injection decisions do not come from the inference thread. The
// caller owns all pacing policy (padding between sentences, per-injection caps,
// cancellation); this is only the override mechanism.
// The text logits the LAST step sampled from, valid until the next step.
//
// The argmax alone cannot tell a confident decision from a coin flip, and the
// difference decides whether a token mismatch against a reference implementation is a
// defect or numerical noise. Two stacks running the same bf16 weights will disagree on
// a 0.3-logit near-tie and agree on everything decided by 2; without the distribution
// there is no way to say which happened. Returns the number of logits, 0 if the graph
// has not run yet.
MOSHI_API int moshi_lm_get_text_logits( moshi_lm_gen_t * gen, std::vector<float> & logits );

MOSHI_API void moshi_lm_personaplex_force_text_token( moshi_lm_gen_t * gen, int token );
MOSHI_API void moshi_lm_personaplex_clear_forced_text_token( moshi_lm_gen_t * gen );
// The armed-but-not-yet-consumed token, or -1.
MOSHI_API int moshi_lm_personaplex_pending_forced_text_token( moshi_lm_gen_t * gen );

MOSHI_API void moshi_lm_send( moshi_lm_gen_t * gen, Entry * entry );
MOSHI_API int moshi_lm_receive( moshi_lm_gen_t * gen, int & text_token, std::vector<int16_t> & audio_tokens );
MOSHI_API void moshi_lm_send2( moshi_lm_gen_t * gen, std::vector<int16_t> & audio_tokens );
MOSHI_API void moshi_lm_receive2( moshi_lm_gen_t * gen, int & text_token, float & vad );
MOSHI_API int moshi_lm_is_active( moshi_lm_gen_t * gen );
MOSHI_API int moshi_lm_is_empty( moshi_lm_gen_t * gen );
MOSHI_API void moshi_lm_machine_reset( moshi_lm_gen_t * gen );

// MARK: Misc


/*int moshi_lm_n_q( moshi_lmmodel_t * lm );
int moshi_lm_max_delay( moshi_lmmodel_t * lm );
int moshi_lm_delay_steps( moshi_lmmodel_t * lm );*/
