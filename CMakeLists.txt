cmake_minimum_required(VERSION 3.26)
project(
        Raytracer
        VERSION 0.0.1
        DESCRIPTION "A simple CUDA raytracer"
        LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
        Eigen
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0)
FetchContent_MakeAvailable(Eigen)

FetchContent_Declare(
        fmtlib
        GIT_PROGRESS TRUE
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 9.1.0)
FetchContent_MakeAvailable(fmtlib)


FetchContent_Declare(
        fmtlib
        GIT_PROGRESS TRUE
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 9.1.0)
FetchContent_MakeAvailable(fmtlib)


FetchContent_Declare(rapidobj
        GIT_REPOSITORY https://github.com/guybrush77/rapidobj.git
        GIT_TAG origin/master)

FetchContent_MakeAvailable(rapidobj)


file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${Raytracer_SOURCE_DIR}/include/raytracer/*.cuh")
add_library(raytracer INTERFACE)
target_sources(raytracer INTERFACE ${headers})

target_include_directories(raytracer INTERFACE "${PROJECT_SOURCE_DIR}/include")
set_target_properties(raytracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(raytracer INTERFACE Eigen3::Eigen fmt::fmt)
source_group(
        TREE "${PROJECT_SOURCE_DIR}/include"
        PREFIX "Header Files"
        FILES ${HEADER_LIST})


add_subdirectory(apps)
