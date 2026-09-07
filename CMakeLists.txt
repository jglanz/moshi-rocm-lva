cmake_minimum_required(VERSION 3.21)
project("moshi" VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(MOSHI_STANDALONE ON)

    # configure project version
    # TODO
else()
    set(MOSHI_STANDALONE OFF)
endif()

include_directories(include)

function(setup_rpath target)
    if (APPLE)
        # On macOS, set RPATH to look in the same directory as the executable
        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "@executable_path"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    else()
        # On Linux, set RPATH to look in the same directory as the executable
        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endfunction()

find_package(Threads REQUIRED)

# ggml exports a real CMake package; consume it instead of cmake/FindGGML.cmake,
# whose optional-backend probes (${GGML_LIBRARY_DIR}/ggml-hip, .../ggml-vulkan) no
# longer match any ggml layout and silently resolve to *-NOTFOUND.
find_package(ggml CONFIG REQUIRED)

# SentencePiece does not export a CMake package; this defines moshi::sentencepiece.
include(moshi-sentencepiece)

option(MOSHI_BUILD_TOOLS    "moshi: build tools"          ${MOSHI_STANDALONE})

# OFF by default, and deliberately so: a consumer that pulls this tree in with
# add_subdirectory(), and the package port that builds it, get no test target and
# no ctest registration unless they ask for one.
option(MOSHI_BUILD_TESTS    "moshi: build tests"          OFF)

add_library(moshi src/moshi.cpp src/json.cpp src/safetensor.cpp)
add_library(moshi::moshi ALIAS moshi)

# Both are PUBLIC: include/moshi/moshi.h includes <ggml.h>, <ggml-backend.h>,
# <ggml-cpu.h> and <sentencepiece_processor.h>, so every consumer needs their
# include directories, and a static libmoshi needs them at final link.
target_link_libraries(moshi PUBLIC ggml::ggml moshi::sentencepiece)
target_link_libraries(moshi PRIVATE Threads::Threads)

target_include_directories(moshi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(moshi PUBLIC cxx_std_20)
setup_rpath(moshi)

if (MOSHI_BUILD_TOOLS)
    add_subdirectory(tools)
endif ()

if (MOSHI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

# ---------------------------------------------------------------------------
# Install / export surface.
#
# Upstream has none at all: no install() and no export() anywhere in CMakeLists.txt,
# tools/CMakeLists.txt or cmake/. Everything below is new.
# ---------------------------------------------------------------------------

install(TARGETS moshi
    EXPORT moshi-targets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/moshi"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT moshi-targets
    FILE moshi-targets.cmake
    NAMESPACE moshi::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/moshi"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/moshi-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/moshi-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/moshi"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/moshi-config-version.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/moshi-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/moshi-config-version.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/moshi-sentencepiece.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/moshi"
)

if (MOSHI_BUILD_TOOLS)
    install(TARGETS
            mimi-encode mimi-decode mimi-play mimi-echo moshi-tts moshi-stt
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
    if (WIN32)
        install(TARGETS moshi-sts RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    endif()
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tools/"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        FILES_MATCHING
            PATTERN "*-config.json"
            PATTERN "kyutai_*.txt"
            PATTERN "Codes4Fun_*.txt"
            PATTERN "moshi-defaults.txt"
    )
endif()
