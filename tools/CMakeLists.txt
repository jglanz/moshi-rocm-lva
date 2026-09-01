
# SDL2 and FFmpeg both export usable CMake packages when they come from vcpkg, so
# cmake/FindFFmpeg.cmake (a pkg-config-driven module that also demands AVFILTER,
# which no tool in this directory includes) is bypassed the same way FindGGML.cmake
# is. FFMPEG (all caps) selects vcpkg's FindFFMPEG.cmake, not the in-tree module.
find_package(SDL2 CONFIG REQUIRED)
find_package(FFMPEG REQUIRED)

set(MOSHI_SDL2 $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>)

function(moshi_tool_ffmpeg target)
    target_include_directories(${target} PRIVATE ${FFMPEG_INCLUDE_DIRS})
    target_link_directories(${target} PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(${target} PRIVATE ${FFMPEG_LIBRARIES})
endfunction()

add_executable(mimi-encode mimi-encode.cpp)
target_link_libraries(mimi-encode PRIVATE moshi::moshi)
moshi_tool_ffmpeg(mimi-encode)
setup_rpath(mimi-encode)

add_executable(mimi-decode mimi-decode.cpp)
target_link_libraries(mimi-decode PRIVATE moshi::moshi)
moshi_tool_ffmpeg(mimi-decode)
setup_rpath(mimi-decode)

add_executable(mimi-play mimi-play.cpp)
target_link_libraries(mimi-play PRIVATE moshi::moshi ${MOSHI_SDL2})
setup_rpath(mimi-play)

add_executable(mimi-echo mimi-echo.cpp)
target_link_libraries(mimi-echo PRIVATE moshi::moshi ${MOSHI_SDL2})
setup_rpath(mimi-echo)

add_executable(moshi-tts moshi-tts.cpp)
target_link_libraries(moshi-tts PRIVATE moshi::moshi ${MOSHI_SDL2})
moshi_tool_ffmpeg(moshi-tts)
setup_rpath(moshi-tts)

add_executable(moshi-stt moshi-stt.cpp)
target_link_libraries(moshi-stt PRIVATE moshi::moshi ${MOSHI_SDL2})
moshi_tool_ffmpeg(moshi-stt)
setup_rpath(moshi-stt)

# moshi-sts is Windows-only at this revision: stt_worker() and translate_worker()
# call _popen/_pclose (moshi-sts.cpp:75,82,145,152) to drive a PowerShell
# System.Speech.Recognition pipeline. It does not compile on Linux/macOS, so it is
# not offered there rather than breaking every other tool in this directory.
if (WIN32)
    add_executable(moshi-sts moshi-sts.cpp)
    target_link_libraries(moshi-sts PRIVATE moshi::moshi ${MOSHI_SDL2})
    moshi_tool_ffmpeg(moshi-sts)
    setup_rpath(moshi-sts)
    set(MOSHI_CONFIG_JSON_TARGET moshi-sts)
else()
    set(MOSHI_CONFIG_JSON_TARGET moshi-tts)
endif()

set(CONFIG_FILES_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB CONFIG_FILES_TO_COPY
    RELATIVE ${CONFIG_FILES_SOURCE_DIR}
    "${CONFIG_FILES_SOURCE_DIR}/*-config.json"
)
set(CONFIG_FILES_DEST_DIR $<TARGET_FILE_DIR:${MOSHI_CONFIG_JSON_TARGET}>)
foreach(file ${CONFIG_FILES_TO_COPY})
    add_custom_command(
        TARGET ${MOSHI_CONFIG_JSON_TARGET}
        POST_BUILD # Run this command after the target's main command
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CONFIG_FILES_SOURCE_DIR}/${file}"
                "${CONFIG_FILES_DEST_DIR}/${file}"
        COMMENT "Copying ${file} to ${CONFIG_FILES_DEST_DIR}"
    )
endforeach()

set(ARIA2_FILES_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ARIA2_FILES_DEST_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(GLOB ARIA2_FILES_TO_COPY
    RELATIVE ${ARIA2_FILES_SOURCE_DIR}
    "${ARIA2_FILES_SOURCE_DIR}/kyutai_*.txt"
    "${ARIA2_FILES_SOURCE_DIR}/Codes4Fun_*.txt"
    "moshi-defaults.txt"
)
add_custom_target(copy_aria2_files ALL
    COMMENT "Creating destination directory for text files"
)
foreach(file ${ARIA2_FILES_TO_COPY})
    add_custom_command(
        TARGET copy_aria2_files
        POST_BUILD # Run this command after the target's main command
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ARIA2_FILES_SOURCE_DIR}/${file}"
                "${ARIA2_FILES_DEST_DIR}/${file}"
        COMMENT "Copying ${file} to ${ARIA2_FILES_DEST_DIR}"
    )
endforeach()
