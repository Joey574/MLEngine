cmake_minimum_required(VERSION 3.18)
project(MLEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# optionally enable cuda
option(USE_CUDA "Enable CUDA support (if available)" ON)

# -----------------------
# Default to a release build
# -----------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Release or Debug)" FORCE)
endif()

# -----------------------
# Detect and enable CUDA if found
# -----------------------
set(USE_CUDA_FOUND FALSE)
if(USE_CUDA)
  find_program(NVCC_FOUND nvcc)
  if(NVCC_FOUND)
    enable_language(CUDA)
    set(USE_CUDA_FOUND TRUE)
    message(STATUS "CUDA compiler found: ${NVCC_FOUND} -- enabling CUDA")
  else()
    set(USE_CUDA_FOUND FALSE)
    message(WARNING "CUDA requested (USE_CUDA=ON) but nvcc not found. Building without CUDA.")
  endif()
else()
  message(STATUS "USE_CUDA=OFF -> Building without CUDA support.")
endif()

# -----------------------
# Optionally enable ccache
# -----------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# -----------------------
# Grab CXX and CUDA source files
# -----------------------
file(GLOB_RECURSE CXX_SOURCE_FILES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCE_FILES CONFIGURE_DEPENDS src/*.cu)

set(SOURCE_FILES ${CXX_SOURCE_FILES})
if(USE_CUDA_FOUND)
  list(APPEND SOURCE_FILES ${CUDA_SOURCE_FILES})
endif()


# create executable
add_executable(MLEngine ${SOURCE_FILES})

# -----------------------
# Build flags
# -----------------------
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(DEBUG_FLAGS_CXX -march=native -O0 -g -DDEBUG -rdynamic -fopenmp -lopenblas)
  set(DEBUG_FLAGS_CUDA -march=native -O0 --extended-lambda)

  target_compile_options(MLEngine PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${DEBUG_FLAGS_CXX}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${USE_CUDA_FOUND}>>:${DEBUG_FLAGS_CUDA}>
  )
else()
  set(RELEASE_FLAGS_CXX -march=native -mtune=native -DNDEBUG -Ofast -fopenmp -lopenblas)
  set(RELEASE_FLAGS_CUDA -march=native -mtune=native -O3 --extended-lambda)

  target_compile_options(MLEngine PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${RELEASE_FLAGS_CXX}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${USE_CUDA_FOUND}>>:${RELEASE_FLAGS_CUDA}>
  )
endif()

# -----------------------
# Required CPU packages
# -----------------------
find_package(yaml-cpp QUIET)
find_package(OpenMP QUIET)
find_package(BLAS QUIET)
target_link_libraries(MLEngine PRIVATE OpenMP::OpenMP_CXX yaml-cpp ${BLAS_LIBRARIES})

# Precompiled header
target_precompile_headers(MLEngine PRIVATE src/Dependencies/pch.h)


# Include FetchContent module for downloading dependencies
include(FetchContent)
  if(USE_CUDA_FOUND)
  # Fetch the official parrot library from NVlabs/parrot
  FetchContent_Declare(
    parrot
    GIT_REPOSITORY https://github.com/NVlabs/parrot.git
    GIT_TAG main
    GIT_SHALLOW TRUE
  )

  # Make parrot available
  FetchContent_MakeAvailable(parrot)

  # Add parrot include directory from FetchContent
  FetchContent_GetProperties(parrot)
  if(parrot_POPULATED)
      include_directories(${parrot_SOURCE_DIR})
      message(STATUS "Using parrot library from: ${parrot_SOURCE_DIR}")

      # Get CCCL from parrot's build and prioritize it over system headers
      # This ensures we use Thrust 3.2.0+ instead of system Thrust 3.0.1
      FetchContent_GetProperties(cccl)
      if(cccl_POPULATED)
          include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR})
          include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/cub)
          include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/thrust)
          include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/libcudacxx/include)

          # Add compiler flags to prioritize CCCL headers
          set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}")
          set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/cub")
          set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/thrust")
          set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/libcudacxx/include")

          message(STATUS "Using CCCL (Thrust 3.2.0+) from: ${cccl_SOURCE_DIR}")
      endif()
  endif()
endif()
