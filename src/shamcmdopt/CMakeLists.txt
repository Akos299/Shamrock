cmake_minimum_required(VERSION 3.9)

project(Shamcmdopt CXX C)

set(Sources
        src/details/generic_opts.cpp
        src/cmdopt.cpp
        src/env.cpp
        src/term_colors.cpp
        src/tty.cpp
    )

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamcmdopt target (Shared lib)")
add_library(shamcmdopt SHARED ${Sources})
else()
message("-- adding shamcmdopt target (object lib)")
add_library(shamcmdopt OBJECT ${Sources})
endif()

target_link_libraries(shamcmdopt PUBLIC shambase)

target_include_directories(shamcmdopt PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
