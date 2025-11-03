# OpenBLASFetch.cmake
FetchContent_Declare(
    OpenBLAS
    GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
    GIT_TAG v0.3.30
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(openblas)
