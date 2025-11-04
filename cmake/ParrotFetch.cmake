# ParrotFetch.cmake
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
