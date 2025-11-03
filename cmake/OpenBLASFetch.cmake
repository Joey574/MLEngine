# OpenBLASFetch.cmake
include(FetchContent)
find_package(OpenBLAS QUIET)

set(NO_SHARED TRUE)
set(USE_OPENMP TRUE)
set(ONLY_CBLAS TRUE)

set(BUILD_DOUBLE OFF)
set(BUILD_COMPLEX OFF)

if(NOT OpenBLAS_FOUND AND NOT TARGET OpenBLAS::OpenBLAS)
    message(STATUS "Fetching OpenBLAS")
    #set(FETCHCONTENT_QUIET TRUE)

    FetchContent_Declare(
        OpenBLAS
        GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
        GIT_TAG develop
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(OpenBLAS)
endif()
