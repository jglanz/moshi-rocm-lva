# moshi — CPU test targets.
#
# Reached only when MOSHI_BUILD_TESTS is ON, which it is not by default: a
# consumer that pulls this tree in with add_subdirectory(), and the package port
# that builds it, are unaffected by everything here.
#
# No test framework and no new dependency: these targets assert with plain code
# and a non-zero exit status, so the tests build anywhere the library does.

# script_grammar_test — the scripted-decoding grammar.
#
# It compiles include/moshi/script_grammar.h directly and links nothing: the
# grammar is header-only and depends on neither ggml nor a model, which is the
# whole point of keeping it separable from the decoder that consults it.
add_executable(script_grammar_test script_grammar_test.cpp)
target_include_directories(script_grammar_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_features(script_grammar_test PRIVATE cxx_std_20)
target_compile_options(script_grammar_test PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra>)
add_test(NAME script_grammar_test COMMAND script_grammar_test)
