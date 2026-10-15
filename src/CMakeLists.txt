file(GLOB_RECURSE CORE_SOURCES
  CONFIGURE_DEPENDS
  *.cpp
  *.hpp
)
list (FILTER CORE_SOURCES EXCLUDE REGEX "main\\.cpp$")

add_library(mlengine_core STATIC ${CORE_SOURCES})

find_package(PkgConfig REQUIRED)

target_include_directories(mlengine_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

include(FetchContent)

find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)

FetchContent_Declare(
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG v9.0.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glaze)

find_library(OPENBLAS_LIB NAMES openblas REQUIRED)
find_path(
  OPENBLAS_INCLUDE_DIR
  NAMES cblas.h
  PATH_SUFFIXES include/openblas include REQUIRED
)
target_include_directories(mlengine_core PRIVATE ${OPENBLAS_INCLUDE_DIR})

set(RELEASE_FLAGS
  -Ofast
  -march=native
  -mtune=native
  -flto=auto
  -fopenmp
  -DNDEBUG
  -fvect-cost-model=unlimited
  -falign-functions=32
  -falign-loops=32
  -funroll-loops
  -fipa-pta
  -fmodulo-sched
  -fmodulo-sched-allow-regmoves
  -fira-loop-pressure
  -fsched-pressure
  -frename-registers
  -fschedule-insns
  -Wno-unused-result
)

set(DEBUG_FLAGS
  -march=native
  -O0 -g -DDEBUG -rdynamic
  -fno-lto -fopenmp
)

add_executable(mlengine main.cpp)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  target_compile_options(mlengine PRIVATE ${DEBUG_FLAGS})
else()
  target_compile_options(mlengine PRIVATE ${RELEASE_FLAGS})
endif()

target_link_libraries(mlengine PRIVATE mlengine_core)
