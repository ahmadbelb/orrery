file(GLOB ORRERY_UNIT_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu")
foreach(src IN LISTS ORRERY_UNIT_TESTS)
  get_filename_component(t "${src}" NAME_WE)
  orrery_add_application(${t} SOURCES "${src}" OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  add_test(NAME ${t} COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_one.sh ${t} $<TARGET_FILE:${t}>)
  set_tests_properties(${t} PROPERTIES
    SKIP_RETURN_CODE 77
    FAIL_REGULAR_EXPRESSION "\\[FAIL\\]")
endforeach()
