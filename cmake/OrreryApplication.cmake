function(orrery_add_application name)
  cmake_parse_arguments(A "" "OUTPUT_DIRECTORY" "SOURCES;DEFINITIONS;INCLUDES;COMPILE_OPTIONS" ${ARGN})
  if(NOT A_SOURCES)
    message(FATAL_ERROR "orrery_add_application(${name}): no SOURCES given")
  endif()
  add_library(${name} MODULE ${A_SOURCES} $<TARGET_OBJECTS:orrery_interposer>)
  target_include_directories(${name} PRIVATE
    ${ORRERY_INCLUDE_DIR} ${A_INCLUDES} ${SIMGRID_INCLUDE_DIR} ${SIMGRID_INCLUDE_DIR}/smpi)
  target_compile_definitions(${name} PRIVATE
    SMPI_NO_OVERRIDE_MALLOC OPEN_MPI=1 MPIX_CUDA_AWARE_SUPPORT=1 ${A_DEFINITIONS})
  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-include>
    $<$<COMPILE_LANGUAGE:CXX>:${ORRERY_INCLUDE_DIR}/mpi-ext.h>
    $<$<COMPILE_LANGUAGE:CUDA>:--pre-include>
    $<$<COMPILE_LANGUAGE:CUDA>:${ORRERY_INCLUDE_DIR}/mpi-ext.h>
    ${A_COMPILE_OPTIONS})
  target_link_libraries(${name} PRIVATE ${SIMGRID_LIBRARY} CUDA::cudart ${CMAKE_DL_LIBS} m)
  target_link_options(${name} PRIVATE -Wl,-Bsymbolic)
  set_target_properties(${name} PROPERTIES
    PREFIX "" SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  if(A_OUTPUT_DIRECTORY)
    set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${A_OUTPUT_DIRECTORY}")
  endif()
endfunction()
