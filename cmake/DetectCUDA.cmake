# -----------------------
# Detect and enable CUDA if found
# -----------------------

set(USE_CUDA_FOUND FALSE)
if(USE_CUDA)
  find_program(NVCC_FOUND nvcc)
  if(NVCC_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    set(USE_CUDA_FOUND TRUE)
    message(STATUS "CUDA compiler found: ${NVCC_FOUND}")
  else()
    set(USE_CUDA_FOUND FALSE)
    message(WARNING "CUDA requested (USE_CUDA=ON) but nvcc not found. Building without CUDA.")
  endif()
else()
  message(STATUS "Building without CUDA support. (USE_CUDA=OFF)")
endif()
