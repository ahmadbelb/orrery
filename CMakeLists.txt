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

project(orrery VERSION 1.0 LANGUAGES CXX CUDA)

if(DEFINED ENV{SIMGRID_PREFIX})
  set(_sg_default "$ENV{SIMGRID_PREFIX}")
else()
  set(_sg_default "/usr/local")
endif()
set(SIMGRID_PREFIX "${_sg_default}" CACHE PATH "SimGrid installation (built with SMPI)")
find_path(SIMGRID_INCLUDE_DIR smpi/smpi.h HINTS "${SIMGRID_PREFIX}/include")
find_library(SIMGRID_LIBRARY simgrid HINTS "${SIMGRID_PREFIX}/lib" "${SIMGRID_PREFIX}/lib64")
find_program(SMPIRUN smpirun HINTS "${SIMGRID_PREFIX}/bin")
if(NOT SIMGRID_INCLUDE_DIR OR NOT SIMGRID_LIBRARY OR NOT SMPIRUN)
  message(FATAL_ERROR "SimGrid with SMPI not found under ${SIMGRID_PREFIX} "
                      "(need include/smpi/smpi.h, lib/libsimgrid.so, bin/smpirun). "
                      "Set -DSIMGRID_PREFIX=<prefix>; see README.md.")
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
