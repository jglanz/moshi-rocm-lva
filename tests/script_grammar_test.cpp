// The scripted-decoding grammar, row by row.
//
// include/moshi/script_grammar.h is the rule the decoder consults on every step
// while a script is live, and it is pure host code with no ggml and no device
// state -- so it can be, and is, tested on the CPU with no model present. There is
// one case per row of the transition table in that header, plus the boundaries the
// rows do not state on their own: the stall cap firing at exactly the cap, the
// eager end-of-script transition, every end-of-script variant, and an out-of-set
// token being reported rather than absorbed.
//
// Fixed token ids are fixtures, which is allowed here and only here: nothing in
// this file is reachable from a runtime path.

#include "moshi/script_grammar.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using moshi_script::AfterScript;
using moshi_script::Allowed;
using moshi_script::Machine;
using moshi_script::Options;
using moshi_script::State;

// The two ids the model uses for structure. `pad` is the model config's text
// padding id on every real generator; `new_word` is 0 by construction.
const int kPad     = 3;
const int kNewWord = 0;

// Content ids, all well above the byte-fallback range so nothing here collides
// with a structural id.
const int kThe   = 1001;
const int kWa    = 1002;  // "wa"    -- word-initial piece
const int kTer   = 1003;  // "ter"   -- continuation of the same word
const int kStop  = 1004;  // "."     -- a piece that does not start a word

int g_failures = 0;
int g_checks   = 0;
const char * g_case = "";

void check( bool ok, const char * expr, int line ) {
    ++g_checks;
    if ( ! ok ) {
        ++g_failures;
        std::printf( "  FAIL [%s] %s (line %d)\n", g_case, expr, line );
    }
}

#define CHECK( expr ) check( ( expr ), #expr, __LINE__ )

// A three-token script "▁the ▁wa ter": two words, the third piece continuing the
// second. word_start marks the pieces that begin a word.
struct Script {
    std::vector<int>  tokens{ kThe, kWa, kTer };
    std::vector<char> word_start{ 1, 1, 0 };

    const int *  t() const { return tokens.data(); }
    const char * w() const { return word_start.data(); }
    int          n() const { return (int) tokens.size(); }
};

Options defaults() {
    Options o;
    o.max_pad_frames_between_chunks = 40;
    o.after_script = AfterScript::wait_for_new_word;
    o.after_script_frames = 0;
    return o;
}

// Arm and walk to the state under test, asserting nothing: the walk itself is
// covered by its own cases.
Machine armed( const Script & s ) {
    Machine m;
    m.arm( s.n() );
    return m;
}

Machine awaiting_first_piece( const Script & s, const Options & o ) {
    Machine m = armed( s );
    m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o );
    return m;
}

// After the first content token: state in_chunk, cursor 1, and word_start[1] is
// true, so this is a word boundary.
Machine in_chunk_at_boundary( const Script & s, const Options & o ) {
    Machine m = awaiting_first_piece( s, o );
    m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o );
    return m;
}

// ---------------------------------------------------------------------------
// One case per row of the table.

void case_none_is_unconstrained() {
    const Script s;
    const Options o = defaults();
    Machine m;
    CHECK( m.state == State::none );
    CHECK( ! m.constraining() );
    const Allowed a = m.allowed( s.w(), o );
    CHECK( ! a.any() );
    // advance() in a non-constraining state changes nothing and is not a violation.
    CHECK( m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::none );
    CHECK( m.cursor == 0 );
    CHECK( m.steps_since_arm == -1 );
}

void case_armed_allows_pad_and_new_word() {
    const Script s;
    const Options o = defaults();
    Machine m = armed( s );
    CHECK( m.state == State::armed );
    CHECK( m.constraining() );
    CHECK( m.steps_since_arm == 0 );
    CHECK( m.steps_since_first_word == -1 );
    const Allowed a = m.allowed( s.w(), o );
    CHECK( a.pad );
    CHECK( a.new_word );
    CHECK( ! a.piece );
}

void case_armed_pad_holds_and_counts() {
    const Script s;
    const Options o = defaults();
    Machine m = armed( s );
    CHECK( m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::armed );
    CHECK( m.pads_since_chunk == 1 );
    CHECK( m.chunks_started == 0 );
    CHECK( m.steps_since_arm == 1 );
    CHECK( m.steps_since_first_word == -1 );
    CHECK( m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.pads_since_chunk == 2 );
    CHECK( m.steps_since_arm == 2 );
}

void case_armed_new_word_opens_a_chunk() {
    const Script s;
    const Options o = defaults();
    Machine m = armed( s );
    m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::awaiting_piece );
    CHECK( m.chunks_started == 1 );
    CHECK( m.pads_since_chunk == 0 );
    CHECK( m.cursor == 0 );
}

void case_awaiting_piece_allows_only_the_script_piece() {
    const Script s;
    const Options o = defaults();
    Machine m = awaiting_first_piece( s, o );
    const Allowed a = m.allowed( s.w(), o );
    CHECK( a.piece );
    CHECK( ! a.pad );
    CHECK( ! a.new_word );
    CHECK( m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::in_chunk );
    CHECK( m.cursor == 1 );
    CHECK( m.steps_since_first_word == 0 );
}

void case_in_chunk_at_a_word_boundary_allows_pad_and_new_word() {
    const Script s;
    const Options o = defaults();
    Machine m = in_chunk_at_boundary( s, o );
    CHECK( m.state == State::in_chunk );
    CHECK( m.cursor == 1 );
    const Allowed a = m.allowed( s.w(), o );
    CHECK( a.piece );
    CHECK( a.pad );
    CHECK( a.new_word );
}

void case_in_chunk_mid_word_allows_only_the_next_piece() {
    const Script s;
    const Options o = defaults();
    Machine m = in_chunk_at_boundary( s, o );
    // Consume "▁wa": the cursor now points at "ter", which does NOT start a word,
    // so nothing may come between it and the piece before it.
    CHECK( m.advance( kWa, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::in_chunk );
    CHECK( m.cursor == 2 );
    const Allowed a = m.allowed( s.w(), o );
    CHECK( a.piece );
    CHECK( ! a.pad );
    CHECK( ! a.new_word );
    // A pad there is out of set, and the machine says so instead of absorbing it.
    Machine before = m;
    CHECK( ! m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == before.state );
    CHECK( m.cursor == before.cursor );
    CHECK( m.pads_since_chunk == before.pads_since_chunk );
}

void case_in_chunk_content_runs_straight_on() {
    // The corpus's C->C transition: a word may continue, and a new word may begin,
    // with no structural token in between.
    Script s;
    s.tokens     = { kThe, kStop, kWa, kTer };
    s.word_start = { 1, 0, 1, 0 };
    const Options o = defaults();
    Machine m = awaiting_first_piece( s, o );
    CHECK( m.advance( kThe,  kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.advance( kStop, kPad, kNewWord, s.t(), s.w(), o ) );  // continuation
    CHECK( m.state == State::in_chunk );
    CHECK( m.cursor == 2 );
    CHECK( m.chunks_started == 1 );
    CHECK( m.advance( kWa,   kPad, kNewWord, s.t(), s.w(), o ) );  // next word, no new_word
    CHECK( m.state == State::in_chunk );
    CHECK( m.cursor == 3 );
    CHECK( m.chunks_started == 1 );
}

void case_in_chunk_pad_moves_to_between() {
    const Script s;
    const Options o = defaults();
    Machine m = in_chunk_at_boundary( s, o );
    CHECK( m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::between );
    CHECK( m.pads_since_chunk == 1 );  // the first pad after content
    const Allowed a = m.allowed( s.w(), o );
    CHECK( a.pad );
    CHECK( a.new_word );
    CHECK( ! a.piece );
}

void case_between_new_word_opens_the_next_chunk() {
    // One of the two new_word shapes: new_word after a pad run (PAD->NW).
    const Script s;
    const Options o = defaults();
    Machine m = in_chunk_at_boundary( s, o );
    m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.pads_since_chunk == 2 );
    CHECK( m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::awaiting_piece );
    CHECK( m.chunks_started == 2 );
    CHECK( m.pads_since_chunk == 0 );
}

void case_in_chunk_new_word_opens_the_next_chunk() {
    // The other new_word shape, attested 189 times in the corpus: new_word directly
    // after content, with no pad in between (C->NW).
    const Script s;
    const Options o = defaults();
    Machine m = in_chunk_at_boundary( s, o );
    CHECK( m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::awaiting_piece );
    CHECK( m.chunks_started == 2 );
    CHECK( m.pads_since_chunk == 0 );
    CHECK( m.cursor == 1 );
}

void case_free_is_terminal_and_unconstrained() {
    const Script s;
    Options o = defaults();
    o.after_script = AfterScript::free;
    Machine m = awaiting_first_piece( s, o );
    m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kWa,  kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.advance( kTer, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::free );
    CHECK( ! m.constraining() );
    CHECK( ! m.allowed( s.w(), o ).any() );
    // It never reaches done on its own: the consumer ends the cue.
    CHECK( m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::free );
}

void case_done_is_unconstrained() {
    const Script s;
    const Options o = defaults();
    Machine m = awaiting_first_piece( s, o );
    m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kWa,  kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kTer, kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.state == State::after_script );
    CHECK( m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::done );
    CHECK( ! m.constraining() );
    CHECK( ! m.allowed( s.w(), o ).any() );
    const int cursor = m.cursor;
    CHECK( m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::done );
    CHECK( m.cursor == cursor );
}

// ---------------------------------------------------------------------------
// The end of the script.

void case_end_of_script_is_eager() {
    // Consuming the LAST token leaves the tail state in the same step, so no row
    // ever evaluates word_start[n]. The flag array here is exactly n bytes long.
    const Script s;
    const Options o = defaults();
    Machine m = awaiting_first_piece( s, o );
    m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kWa,  kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.state == State::in_chunk );
    CHECK( m.cursor == 2 );
    CHECK( m.advance( kTer, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.cursor == s.n() );
    CHECK( m.state == State::after_script );  // not in_chunk, not one step later
}

void case_after_script_wait_for_new_word() {
    const Script s;
    const Options o = defaults();
    Machine m = awaiting_first_piece( s, o );
    m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kWa,  kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kTer, kPad, kNewWord, s.t(), s.w(), o );
    const Allowed a = m.allowed( s.w(), o );
    CHECK( a.pad );
    CHECK( a.new_word );
    CHECK( ! a.piece );
    CHECK( m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::after_script );
    CHECK( m.pads_since_chunk == 1 );
    CHECK( m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::done );
    // The terminal new_word does not open a chunk.
    CHECK( m.chunks_started == 1 );
}

void case_after_script_pad_frames() {
    const Script s;
    Options o = defaults();
    o.after_script = AfterScript::pad_frames;
    o.after_script_frames = 3;
    Machine m = awaiting_first_piece( s, o );
    m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kWa,  kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kTer, kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.state == State::after_script );
    const Allowed a = m.allowed( s.w(), o );
    CHECK( a.pad );
    CHECK( ! a.new_word );  // {pad} only under a bounded tail
    CHECK( ! a.piece );
    // A new_word in the tail is out of set.
    CHECK( ! m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::after_script );
    for ( int i = 0; i < 3; ++i ) {
        CHECK( m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o ) );
        CHECK( m.state == ( i < 2 ? State::after_script : State::done ) );
    }
    CHECK( m.pads_since_chunk == 3 );
}

void case_after_script_pad_frames_zero_closes_at_once() {
    const Script s;
    Options o = defaults();
    o.after_script = AfterScript::pad_frames;
    o.after_script_frames = 0;
    Machine m = awaiting_first_piece( s, o );
    m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kWa,  kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.advance( kTer, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::done );
}

// ---------------------------------------------------------------------------
// The stall cap.

void case_cap_fires_at_exactly_the_cap() {
    const Script s;
    Options o = defaults();
    o.max_pad_frames_between_chunks = 4;
    Machine m = armed( s );
    for ( int i = 0; i < 3; ++i )
        m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o );
    // One short of the cap: padding is still the model's own choice.
    CHECK( m.pads_since_chunk == 3 );
    Allowed a = m.allowed( s.w(), o );
    CHECK( a.pad );
    CHECK( a.new_word );
    CHECK( m.stalls_capped == 0 );
    CHECK( m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o ) );
    // At the cap: the set collapses to {new_word} and the step is counted.
    CHECK( m.pads_since_chunk == 4 );
    a = m.allowed( s.w(), o );
    CHECK( ! a.pad );
    CHECK( a.new_word );
    CHECK( m.stalls_capped == 0 );
    CHECK( ! m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o ) );  // pad is now out of set
    CHECK( m.stalls_capped == 1 );
    CHECK( m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.stalls_capped == 2 );
    CHECK( m.state == State::awaiting_piece );
    CHECK( m.pads_since_chunk == 0 );
}

void case_cap_governs_between_and_the_terminal_wait() {
    const Script s;
    Options o = defaults();
    o.max_pad_frames_between_chunks = 2;
    Machine m = in_chunk_at_boundary( s, o );
    m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.state == State::between );
    CHECK( ! m.allowed( s.w(), o ).pad );
    m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.stalls_capped == 1 );

    // The same cap bounds the wait after the last token; there is no separate
    // terminal cap.
    m.advance( kWa,  kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kTer, kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.state == State::after_script );
    m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o );
    m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o );
    CHECK( ! m.allowed( s.w(), o ).pad );
    CHECK( m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::done );
    CHECK( m.stalls_capped == 2 );
}

void case_cap_off_never_collapses() {
    const Script s;
    Options o = defaults();
    o.max_pad_frames_between_chunks = 0;  // no cap
    Machine m = armed( s );
    for ( int i = 0; i < 200; ++i )
        CHECK( m.advance( kPad, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.pads_since_chunk == 200 );
    CHECK( m.stalls_capped == 0 );
    CHECK( m.allowed( s.w(), o ).pad );
}

// ---------------------------------------------------------------------------
// Word-start flags.

void case_index_zero_is_a_word_start_whatever_the_flag_says() {
    Script s;
    s.word_start = { 0, 1, 0 };  // a caller that did not normalise index 0
    const Options o = defaults();
    CHECK( Machine::word_start_at( s.w(), 0 ) );
    Machine m = awaiting_first_piece( s, o );
    CHECK( m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::in_chunk );
    CHECK( m.allowed( s.w(), o ).pad );  // word_start[1] is set, so this is a boundary
}

void case_null_word_start_is_one_chunk() {
    const Script s;
    const Options o = defaults();
    Machine m = armed( s );
    m.advance( kNewWord, kPad, kNewWord, s.t(), NULL, o );
    CHECK( m.advance( kThe, kPad, kNewWord, s.t(), NULL, o ) );
    // With no flags only index 0 starts a word, so the whole script runs as one
    // chunk: no pad and no new_word may interrupt it.
    const Allowed a = m.allowed( NULL, o );
    CHECK( a.piece );
    CHECK( ! a.pad );
    CHECK( ! a.new_word );
    CHECK( ! m.advance( kPad,     kPad, kNewWord, s.t(), NULL, o ) );
    CHECK( ! m.advance( kNewWord, kPad, kNewWord, s.t(), NULL, o ) );
    CHECK( m.advance( kWa,  kPad, kNewWord, s.t(), NULL, o ) );
    CHECK( m.advance( kTer, kPad, kNewWord, s.t(), NULL, o ) );
    CHECK( m.state == State::after_script );
    CHECK( m.chunks_started == 1 );
}

// ---------------------------------------------------------------------------
// Out-of-set tokens, counters, and clearing.

void case_out_of_set_tokens_are_reported() {
    const Script s;
    const Options o = defaults();

    // Content where only {pad, new_word} is allowed.
    Machine m = armed( s );
    CHECK( ! m.advance( kThe, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m.state == State::armed );
    CHECK( m.cursor == 0 );
    CHECK( m.steps_since_arm == 1 );  // the step still happened

    // A pad where the next piece is owed.
    Machine m2 = awaiting_first_piece( s, o );
    CHECK( ! m2.advance( kPad, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m2.state == State::awaiting_piece );
    CHECK( ! m2.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m2.chunks_started == 1 );  // not a second chunk

    // The WRONG content token where the script's own piece is owed.
    CHECK( ! m2.advance( kWa, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m2.cursor == 0 );
    CHECK( m2.advance( kThe, kPad, kNewWord, s.t(), s.w(), o ) );
    CHECK( m2.cursor == 1 );
}

void case_counters_track_the_walk() {
    const Script s;
    const Options o = defaults();
    Machine m = armed( s );
    CHECK( m.steps_since_arm == 0 );
    CHECK( m.steps_since_first_word == -1 );
    m.advance( kPad,     kPad, kNewWord, s.t(), s.w(), o );  // 1
    m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o );  // 2
    CHECK( m.steps_since_first_word == -1 );
    m.advance( kThe,     kPad, kNewWord, s.t(), s.w(), o );  // 3, first content
    CHECK( m.steps_since_arm == 3 );
    CHECK( m.steps_since_first_word == 0 );
    m.advance( kPad,     kPad, kNewWord, s.t(), s.w(), o );  // 4
    CHECK( m.steps_since_arm == 4 );
    CHECK( m.steps_since_first_word == 1 );
    CHECK( m.pads_since_chunk == 1 );
    m.advance( kNewWord, kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.pads_since_chunk == 0 );
    m.advance( kWa,      kPad, kNewWord, s.t(), s.w(), o );
    CHECK( m.pads_since_chunk == 0 );
    CHECK( m.chunks_started == 2 );
}

void case_clear_returns_to_none() {
    const Script s;
    const Options o = defaults();
    Machine m = in_chunk_at_boundary( s, o );
    m.clear();
    CHECK( m.state == State::none );
    CHECK( ! m.constraining() );
    CHECK( m.cursor == 0 );
    CHECK( m.n == 0 );
    CHECK( m.pads_since_chunk == 0 );
    CHECK( m.chunks_started == 0 );
    CHECK( m.stalls_capped == 0 );
    CHECK( m.steps_since_arm == -1 );
    CHECK( m.steps_since_first_word == -1 );
}

struct Case {
    const char * name;
    void ( *fn )();
};

const Case kCases[] = {
    { "none_is_unconstrained",                        case_none_is_unconstrained },
    { "armed_allows_pad_and_new_word",                case_armed_allows_pad_and_new_word },
    { "armed_pad_holds_and_counts",                   case_armed_pad_holds_and_counts },
    { "armed_new_word_opens_a_chunk",                 case_armed_new_word_opens_a_chunk },
    { "awaiting_piece_allows_only_the_script_piece",  case_awaiting_piece_allows_only_the_script_piece },
    { "in_chunk_at_a_word_boundary",                  case_in_chunk_at_a_word_boundary_allows_pad_and_new_word },
    { "in_chunk_mid_word",                            case_in_chunk_mid_word_allows_only_the_next_piece },
    { "in_chunk_content_runs_straight_on",            case_in_chunk_content_runs_straight_on },
    { "in_chunk_pad_moves_to_between",                case_in_chunk_pad_moves_to_between },
    { "between_new_word_opens_the_next_chunk",        case_between_new_word_opens_the_next_chunk },
    { "in_chunk_new_word_opens_the_next_chunk",       case_in_chunk_new_word_opens_the_next_chunk },
    { "free_is_terminal_and_unconstrained",           case_free_is_terminal_and_unconstrained },
    { "done_is_unconstrained",                        case_done_is_unconstrained },
    { "end_of_script_is_eager",                       case_end_of_script_is_eager },
    { "after_script_wait_for_new_word",               case_after_script_wait_for_new_word },
    { "after_script_pad_frames",                      case_after_script_pad_frames },
    { "after_script_pad_frames_zero_closes_at_once",  case_after_script_pad_frames_zero_closes_at_once },
    { "cap_fires_at_exactly_the_cap",                 case_cap_fires_at_exactly_the_cap },
    { "cap_governs_between_and_the_terminal_wait",    case_cap_governs_between_and_the_terminal_wait },
    { "cap_off_never_collapses",                      case_cap_off_never_collapses },
    { "index_zero_is_a_word_start",                   case_index_zero_is_a_word_start_whatever_the_flag_says },
    { "null_word_start_is_one_chunk",                 case_null_word_start_is_one_chunk },
    { "out_of_set_tokens_are_reported",               case_out_of_set_tokens_are_reported },
    { "counters_track_the_walk",                      case_counters_track_the_walk },
    { "clear_returns_to_none",                        case_clear_returns_to_none },
};

} // namespace

int main( int argc, char ** argv ) {
    const char * only = argc > 1 ? argv[1] : NULL;
    int ran = 0;
    for ( const Case & c : kCases ) {
        if ( only && std::strcmp( only, c.name ) != 0 )
            continue;
        const int before = g_failures;
        g_case = c.name;
        c.fn();
        ++ran;
        std::printf( "%s %s\n", g_failures == before ? "ok  " : "FAIL", c.name );
    }
    if ( ran == 0 ) {
        std::printf( "script_grammar_test: no case matched \"%s\"\n", only ? only : "" );
        return 2;
    }
    std::printf( "script_grammar_test: %d case(s), %d check(s), %d failure(s)\n",
                 ran, g_checks, g_failures );
    return g_failures == 0 ? 0 : 1;
}
