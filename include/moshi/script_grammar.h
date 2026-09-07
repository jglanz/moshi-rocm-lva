#pragma once

// Scripted (constrained) text decoding -- the grammar, as a pure state machine.
//
// WHAT THIS IS. A caller that wants the model to SAY a specific answer, in the
// model's own voice and on the model's own timing, supplies the answer as text
// token ids. This header is the rule that says which token ids may be chosen on
// each step while such a script is live. It holds no logits, no tensors and no
// device state: the decoder reads the text logits it already computes, asks this
// machine which candidates are legal, picks the highest-scoring one among them,
// and reports the choice back here.
//
// WHY A CANDIDATE SET AND NOT A FORCED TOKEN. Forcing a token id per frame makes
// the caller own the pacing -- when each word starts, how long each word is held,
// where the pauses fall -- and the model then voices a rhythm nobody measured.
// Constraining the candidate set leaves every timing decision with the model: it
// still chooses, every frame, between "hold" (pad), "start the next chunk"
// (new_word) and "emit the next piece"; only the CONTENT of the piece is ours.
//
// HEADER-ONLY AND DEPENDENCY-FREE on purpose: the same code that runs inside the
// decoder is the code a host-side test compiles, so the test scores the shipping
// implementation rather than a paraphrase of it.
//
// ---------------------------------------------------------------------------
// THE GRAMMAR IS THE MODEL'S OWN, NOT AN INVENTED ONE.
//
// It was derived from 26 recorded raw text-token taps of free generation on this
// model (12 276 rows). Symbol census: pad 9983, new_word 892, content 1401.
// Transition counts between consecutive emitted symbols (PAD = the text padding
// id, NW = new_word (id 0), C = any content id):
//
//     PAD->PAD  9260      NW->C     881
//     PAD->NW    698      NW->PAD     5
//     PAD->C       1      NW->NW      5
//     C->PAD     692      C->C      519
//     C->NW      189
//
// What that data says, and what this machine therefore allows:
//
//   * new_word MAY follow a content token with no pad in between (C->NW, 189).
//   * A word may continue straight through a content token (C->C, 519): the
//     pieces of one word are consecutive, and a word may also begin right after
//     the previous word's last piece.
//   * After new_word the next symbol is content 881 times out of 891, so a
//     chunk that has just opened MUST emit content. The 10 native NW->PAD /
//     NW->NW cases are deliberately forbidden: a chunk that opens and then
//     stalls would trip the stall cap, which would force a second new_word --
//     exactly the NW->NW loop.
//   * Content after a pad happens once in 699 pad-run terminations (PAD->C = 1),
//     so no pad is allowed in the middle of a word.
//
// The three native shapes this grammar forbids are therefore PAD->C (1),
// NW->PAD (5) and NW->NW (5): 11 of roughly 12 250 observed transitions.
//
// There is no end-of-text token on this stream -- the only ids below the content
// range the model ever emits are new_word (0) and the pad id -- so "the model has
// finished that word" is signalled by the model's own new_word, which is what the
// default end-of-script rule waits for.
//
// ---------------------------------------------------------------------------
// THE TRANSITION TABLE. The state is the PREVIOUS emitted symbol, because that is
// the shape of the data above. `c` is `cursor`, `n` is the script length,
// `word_start[i]` is the caller's flag marking a piece that begins a word (index 0
// is normalised to true; a NULL array means only index 0 is a word start, i.e. the
// whole script is one chunk).
//
//   state             | allowed set                                   | on pad                        | on new_word            | on script[c]
//   ------------------+-----------------------------------------------+-------------------------------+------------------------+---------------------------------
//   none              | no constraint (the decoder skips the branch)   | --                            | --                     | --
//   armed             | {pad, new_word}                               | stay, pads++ (cap -> {new_word}) | -> awaiting_piece   | --
//   awaiting_piece    | {script[c]}                                   | --                            | --                     | consume -> in_chunk
//   in_chunk          | {script[c], pad, new_word} if word_start[c],   | -> between                    | -> awaiting_piece      | consume, stay
//                     |   else {script[c]}                            |   (word boundary only)        |                        |
//   between           | {pad, new_word}                               | stay, pads++ (cap -> {new_word}) | -> awaiting_piece   | --
//   after_script      | wait_for_new_word: {pad, new_word}            | stay, pads++ (cap -> {new_word}) | -> done             | --
//                     | pad_frames N:      {pad} for N steps          | pads++, -> done after N       | --                     | --
//   free              | no constraint, terminal (never reaches done)   | --                            | --                     | --
//   done              | no constraint, terminal                       | --                            | --                     | --
//
// EAGERNESS AT THE END OF THE SCRIPT. Consuming script[n-1] leaves the machine in
// after_script (or free, or done) IN THE SAME STEP. Nothing ever evaluates
// word_start[n]; there is no state in which the cursor sits at n with the script
// still constraining.
//
// COUNTERS.
//   * chunks_started counts every transition INTO awaiting_piece while a script is
//     live -- i.e. every new_word the machine accepted. One definition, so an
//     external observer counting new_word symbols on the raw tap gets the same
//     number.
//   * pads_since_chunk is the length of the CURRENT pad run: 1 on the first pad
//     after content, incremented by each further pad, reset to 0 by new_word or by
//     content. The stall cap is checked against it.
//   * stalls_capped counts the steps on which the cap collapsed the allowed set to
//     {new_word}. A capped step is a step whose timing the CALLER chose, not the
//     model, which is why the count is reported rather than hidden.
//   * steps_since_arm is 0 at the moment the script is armed and is incremented
//     once per advance(); steps_since_first_word stays -1 until the step that
//     chooses the first content token, which sets it to 0.
//
// THE STALL CAP. max_pad_frames_between_chunks governs armed, between and
// after_script under wait_for_new_word. It exists because a model told to stay
// quiet will choose pad forever: without it an armed script could never start and
// never finish. <= 0 disables it entirely, which is only safe for a caller that
// has its own way out.
//
// VIOLATIONS. advance() returns false if the chosen token was outside the allowed
// set, and then changes nothing but the step counters. Under the decoder that
// cannot happen -- the choice is made FROM the allowed set -- which is exactly why
// the count of false returns is the proof that the mechanism ran.

namespace moshi_script {

// The previous emitted symbol, which is what the allowed set is keyed on. The
// integer values are part of the public progress surface; do not renumber.
enum class State : int {
    none           = 0,  // no script; the decoder's own sampled token stands
    armed          = 1,  // a script is set, nothing emitted for it yet
    awaiting_piece = 2,  // last symbol was new_word: the next piece is owed
    in_chunk       = 3,  // last symbol was content
    between        = 4,  // last symbol was pad, mid-script
    after_script   = 5,  // every script token has been chosen; the tail rule runs
    free           = 6,  // script finished under `free`: terminal, unconstrained
    done           = 7,  // script finished and closed: terminal, unconstrained
};

// What happens once the last script token has been chosen.
enum class AfterScript : int {
    wait_for_new_word = 0,  // hold until the model says "that word is finished"
    pad_frames        = 1,  // a bounded masked tail: {pad} for N steps, then done
    free              = 2,  // lift the constraint and let the caller close the cue
};

struct Options {
    // Longest run of pads allowed before the set collapses to {new_word}.
    // <= 0 means no cap.
    int max_pad_frames_between_chunks = 40;
    AfterScript after_script = AfterScript::wait_for_new_word;
    // Length of the tail under AfterScript::pad_frames.
    int after_script_frames = 0;
};

// The candidate set for one step. `piece` means "the script token at the cursor";
// the caller reads that id out of its own script array.
struct Allowed {
    bool pad      = false;
    bool new_word = false;
    bool piece    = false;

    bool any() const { return pad || new_word || piece; }
};

struct Machine {
    State state = State::none;
    int cursor = 0;                  // == n once every script token was chosen
    int n = 0;
    int pads_since_chunk = 0;
    int chunks_started = 0;
    int stalls_capped = 0;
    int steps_since_arm = -1;        // -1 until armed
    int steps_since_first_word = -1; // -1 until the first content token

    // index 0 is a word start by definition; a NULL array means ONLY index 0 is,
    // i.e. the whole script is a single chunk.
    static bool word_start_at( const char * word_start, int i ) {
        if ( i <= 0 )
            return true;
        if ( ! word_start )
            return false;
        return word_start[i] != 0;
    }

    void clear() {
        state = State::none;
        cursor = 0;
        n = 0;
        pads_since_chunk = 0;
        chunks_started = 0;
        stalls_capped = 0;
        steps_since_arm = -1;
        steps_since_first_word = -1;
    }

    // Arm a script of n_tokens tokens. n_tokens must be >= 1: an empty script has
    // nothing to constrain, and the caller rejects it before it gets here.
    void arm( int n_tokens ) {
        clear();
        state = State::armed;
        n = n_tokens;
        steps_since_arm = 0;
    }

    // True while the machine has a say in the next token. In the other states the
    // decoder skips the whole branch and its own sampled token stands.
    bool constraining() const {
        return state == State::armed
            || state == State::awaiting_piece
            || state == State::in_chunk
            || state == State::between
            || state == State::after_script;
    }

    // True when the current pad run has reached the cap, so this step's set
    // collapses to {new_word}. Only the pad-holding states consult it.
    bool cap_reached( const Options & opts ) const {
        return opts.max_pad_frames_between_chunks > 0
            && pads_since_chunk >= opts.max_pad_frames_between_chunks;
    }

    Allowed allowed( const char * word_start, const Options & opts ) const {
        Allowed a;
        switch ( state ) {
        case State::armed:
        case State::between:
            a.new_word = true;
            a.pad      = ! cap_reached( opts );
            break;
        case State::awaiting_piece:
            a.piece = cursor < n;
            break;
        case State::in_chunk:
            a.piece = cursor < n;
            // A pad or a chunk start may only land on a word boundary: the pieces
            // of one word are consecutive (PAD->C = 1 in the corpus above).
            if ( cursor < n && word_start_at( word_start, cursor ) ) {
                a.pad      = true;
                a.new_word = true;
            }
            break;
        case State::after_script:
            if ( opts.after_script == AfterScript::pad_frames ) {
                a.pad = true;
            } else {
                a.new_word = true;
                a.pad      = ! cap_reached( opts );
            }
            break;
        case State::none:
        case State::free:
        case State::done:
            break;
        }
        return a;
    }

    // Record the token the decoder chose. Returns false -- and changes nothing but
    // the step counters -- if it was outside the allowed set.
    //
    // Ties are the caller's to break, but the order this checks in is the order the
    // decoder uses when two candidate ids collide: script[c], then new_word, then
    // pad.
    bool advance( int chosen, int pad_id, int new_word_id,
                  const int * script, const char * word_start,
                  const Options & opts ) {
        if ( ! constraining() )
            return true;

        const Allowed a = allowed( word_start, opts );
        const bool capped = cap_reached( opts )
            && ( state == State::armed
              || state == State::between
              || ( state == State::after_script
                   && opts.after_script != AfterScript::pad_frames ) );
        const bool first_word_pending = steps_since_first_word < 0;

        bool ok = true;
        bool content = false;

        switch ( state ) {
        case State::armed:
        case State::between:
            if ( a.new_word && chosen == new_word_id ) {
                start_chunk();
            } else if ( a.pad && chosen == pad_id ) {
                ++pads_since_chunk;
            } else {
                ok = false;
            }
            break;

        case State::awaiting_piece:
            if ( a.piece && chosen == script[cursor] ) {
                consume( opts );
                content = true;
            } else {
                ok = false;
            }
            break;

        case State::in_chunk:
            if ( a.piece && chosen == script[cursor] ) {
                consume( opts );
                content = true;
            } else if ( a.new_word && chosen == new_word_id ) {
                start_chunk();
            } else if ( a.pad && chosen == pad_id ) {
                ++pads_since_chunk;
                state = State::between;
            } else {
                ok = false;
            }
            break;

        case State::after_script:
            if ( opts.after_script == AfterScript::pad_frames ) {
                if ( chosen == pad_id ) {
                    ++pads_since_chunk;
                    if ( pads_since_chunk >= opts.after_script_frames )
                        state = State::done;
                } else {
                    ok = false;
                }
            } else {
                if ( a.new_word && chosen == new_word_id ) {
                    pads_since_chunk = 0;
                    state = State::done;
                } else if ( a.pad && chosen == pad_id ) {
                    ++pads_since_chunk;
                } else {
                    ok = false;
                }
            }
            break;

        case State::none:
        case State::free:
        case State::done:
            break;
        }

        if ( capped )
            ++stalls_capped;
        if ( steps_since_arm >= 0 )
            ++steps_since_arm;
        if ( content && first_word_pending ) {
            steps_since_first_word = 0;
        } else if ( steps_since_first_word >= 0 ) {
            ++steps_since_first_word;
        }
        return ok;
    }

private:
    // Every chunk begins at a new_word, and this is the only place that counts one.
    void start_chunk() {
        state = State::awaiting_piece;
        pads_since_chunk = 0;
        ++chunks_started;
    }

    // Consume script[cursor]. The end of the script is taken EAGERLY: consuming the
    // last token leaves the tail state in this same step, so word_start[n] is never
    // read.
    void consume( const Options & opts ) {
        ++cursor;
        pads_since_chunk = 0;
        if ( cursor < n ) {
            state = State::in_chunk;
            return;
        }
        switch ( opts.after_script ) {
        case AfterScript::free:
            state = State::free;
            break;
        case AfterScript::pad_frames:
            state = opts.after_script_frames > 0 ? State::after_script : State::done;
            break;
        case AfterScript::wait_for_new_word:
            state = State::after_script;
            break;
        }
    }
};

} // namespace moshi_script
