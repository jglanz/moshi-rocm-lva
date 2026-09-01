# Locate SentencePiece and expose it as the imported target moshi::sentencepiece.
#
# Upstream SentencePiece ships NO CMake package config — only headers, the libraries
# and a pkg-config module. (Verified against the vcpkg port: `vcpkg install
# sentencepiece` prints "sentencepiece provides pkg-config modules: sentencepiece",
# and vcpkg_installed/<triplet>/share/sentencepiece contains no *-config.cmake.)
# So this cannot be a find_package(sentencepiece CONFIG) and must not pretend to be.
#
# This file is installed next to moshi-config.cmake and included from it, so the
# consumer side resolves exactly the same way the build side did.

if(NOT TARGET moshi::sentencepiece)
    find_path(MOSHI_SENTENCEPIECE_INCLUDE_DIR
        NAMES sentencepiece_processor.h
        HINTS ${SentencePiece_INCLUDE_DIR} ${SentencePiece_ROOT}
        DOC "SentencePiece include directory")

    find_library(MOSHI_SENTENCEPIECE_LIBRARY
        NAMES sentencepiece
        HINTS ${SentencePiece_LIBRARY_DIR} ${SentencePiece_ROOT}
        DOC "SentencePiece library")

    if(NOT MOSHI_SENTENCEPIECE_INCLUDE_DIR OR NOT MOSHI_SENTENCEPIECE_LIBRARY)
        message(FATAL_ERROR
            "moshi: SentencePiece not found. Provide it on CMAKE_PREFIX_PATH, or set "
            "SentencePiece_INCLUDE_DIR / SentencePiece_LIBRARY_DIR. "
            "(include: ${MOSHI_SENTENCEPIECE_INCLUDE_DIR}, library: ${MOSHI_SENTENCEPIECE_LIBRARY})")
    endif()

    mark_as_advanced(MOSHI_SENTENCEPIECE_INCLUDE_DIR MOSHI_SENTENCEPIECE_LIBRARY)

    add_library(moshi::sentencepiece UNKNOWN IMPORTED)
    set_target_properties(moshi::sentencepiece PROPERTIES
        IMPORTED_LOCATION "${MOSHI_SENTENCEPIECE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MOSHI_SENTENCEPIECE_INCLUDE_DIR}")

    # A STATIC libsentencepiece.a leaves protobuf-lite (and, through it, Abseil)
    # symbols unresolved — sentencepiece.pc declares `Requires: protobuf-lite`.
    # Protobuf DOES export a CMake package, so use it for the transitive closure
    # instead of re-deriving link lines from pkg-config.
    get_filename_component(_moshi_sp_suffix "${MOSHI_SENTENCEPIECE_LIBRARY}" LAST_EXT)
    if(_moshi_sp_suffix STREQUAL "${CMAKE_STATIC_LIBRARY_SUFFIX}")
        find_package(protobuf CONFIG QUIET)
        if(TARGET protobuf::libprotobuf-lite)
            set_property(TARGET moshi::sentencepiece APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES protobuf::libprotobuf-lite)
        endif()
    endif()
    unset(_moshi_sp_suffix)
endif()
