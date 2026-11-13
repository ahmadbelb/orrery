cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED CMAKE_CXX_COMPILER
   AND NOT DEFINED ENV{CXX} AND NOT DEFINED ENV{CUDAHOSTCXX})
  foreach(v 13 12 11)
    find_program(ORRERY_HOST_CXX g++-${v})
    if(ORRERY_HOST_CXX)
      set(CMAKE_CXX_COMPILER "${ORRERY_HOST_CXX}" CACHE FILEPATH "C++ compiler")
      set(CMAKE_CUDA_HOST_COMPILER "${ORRERY_HOST_CXX}" CACHE FILEPATH "host compiler nvcc calls")
      break()
    endif()
  endforeach()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
  find_program(ORRERY_NVIDIA_SMI nvidia-smi)
  if(ORRERY_NVIDIA_SMI)
    execute_process(COMMAND ${ORRERY_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE _cc RESULT_VARIABLE _rc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _cc "${_cc}")
    if(_rc EQUAL 0 AND _cc)
      string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "${_cc}")
      message(STATUS "CUDA architecture ${CMAKE_CUDA_ARCHITECTURES} from the local GPU (nvidia-smi)")
    endif()
  endif()
endif()
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  find_program(ORRERY_NVCC nvcc HINTS ENV CUDA_HOME ENV CUDA_PATH PATH_SUFFIXES bin PATHS /usr/local/cuda/bin)
  if(NOT ORRERY_NVCC)
    message(FATAL_ERROR "nvcc not found. Orrery needs the CUDA toolkit to compile CUDA code "
                        "(no GPU or driver needed). Install it, or set CUDA_HOME to its folder; see README.md.")
  endif()
  set(CMAKE_CUDA_COMPILER "${ORRERY_NVCC}" CACHE FILEPATH "CUDA compiler")
endif()

project(orrery VERSION 1.0 LANGUAGES CXX CUDA)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12 OR CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
  message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} found (${CMAKE_CUDA_COMPILER}). Orrery needs CUDA 12 "
                      "(tested with 12.4). Install it, set CUDA_HOME to its folder and use a new build folder; see README.md.")
endif()

set(SIMGRID_PREFIX "" CACHE PATH "SimGrid installation (built with SMPI)")
set(_sg ${SIMGRID_PREFIX} $ENV{SIMGRID_PREFIX})
find_program(SMPIRUN smpirun HINTS ${_sg} PATH_SUFFIXES bin)
if(SMPIRUN)
  get_filename_component(_sg_root "${SMPIRUN}" DIRECTORY)
  get_filename_component(_sg_root "${_sg_root}" DIRECTORY)
  list(APPEND _sg "${_sg_root}")
endif()
find_path(SIMGRID_INCLUDE_DIR smpi/smpi.h HINTS ${_sg} PATH_SUFFIXES include)
find_library(SIMGRID_LIBRARY simgrid HINTS ${_sg} PATH_SUFFIXES lib lib64)
if(NOT SIMGRID_INCLUDE_DIR OR NOT SIMGRID_LIBRARY OR NOT SMPIRUN)
  message(FATAL_ERROR "SimGrid with SMPI not found (need include/smpi/smpi.h, lib/libsimgrid.so, bin/smpirun). "
                      "Put its bin folder on PATH, or set SIMGRID_PREFIX to the folder you installed it in; see README.md.")
endif()
find_package(CUDAToolkit REQUIRED)
message(STATUS "SimGrid: ${SIMGRID_LIBRARY}  smpirun: ${SMPIRUN}")
message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}  C++: ${CMAKE_CXX_COMPILER}")

add_library(orrery_interposer OBJECT src/orrery_interpose.cpp src/transfer_model.cpp)
target_include_directories(orrery_interposer PRIVATE
  ${SIMGRID_INCLUDE_DIR} ${SIMGRID_INCLUDE_DIR}/smpi)
set_target_properties(orrery_interposer PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(ORRERY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
include(cmake/OrreryApplication.cmake)

orrery_add_application(simple_smpi
  SOURCES examples/simple/simple.cu
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

enable_testing()
add_test(NAME smoke COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.sh $<TARGET_FILE:simple_smpi>)
add_subdirectory(tests)

add_custom_target(example
  COMMAND ${CMAKE_COMMAND} -E env SIMPLE_SO=$<TARGET_FILE:simple_smpi> bash ${CMAKE_CURRENT_SOURCE_DIR}/examples/simple/run.sh
  DEPENDS simple_smpi USES_TERMINAL
  COMMENT "simple example on 4 simulated GPUs")
