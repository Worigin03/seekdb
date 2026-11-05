add_executable(ai_function_tests EXCLUDE_FROM_ALL
  sql/engine/expr/test_ai_function_enhancement.cpp)
target_link_libraries(ai_function_tests PRIVATE
  ob_base oceanbase
  "${DEP_DIR}/lib/libgtest_main.a" "${DEP_DIR}/lib/libgtest.a" pthread)
target_include_directories(ai_function_tests PRIVATE "${DEP_DIR}/include")
enable_testing()
add_test(NAME ai_function_tests COMMAND ai_function_tests)

add_executable(file_scan_tests EXCLUDE_FROM_ALL
  sql/engine/basic/test_file_scan_utils.cpp
  ../src/sql/engine/basic/file_scan_utils.cpp
  ../src/sql/engine/basic/parquet_reader.cpp)
target_include_directories(file_scan_tests PRIVATE "${CMAKE_SOURCE_DIR}/src" "${DEP_DIR}/include")
target_link_libraries(file_scan_tests PRIVATE
  "${DEP_DIR}/lib/libgtest_main.a" "${DEP_DIR}/lib/libgtest.a" pthread)
add_test(NAME file_scan_tests COMMAND file_scan_tests)
set_tests_properties(file_scan_tests PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(file_sql_parser_tests EXCLUDE_FROM_ALL
  sql/parser/test_file_sql_parser.cpp sql/parser/test_ai_column_parser.cpp)
target_include_directories(file_sql_parser_tests PRIVATE "${DEP_DIR}/include")
target_link_libraries(file_sql_parser_tests PRIVATE ob_base oceanbase
  "${DEP_DIR}/lib/libgtest_main.a" "${DEP_DIR}/lib/libgtest.a" pthread)
add_test(NAME file_sql_parser_tests COMMAND file_sql_parser_tests)

add_executable(ai_column_schema_tests EXCLUDE_FROM_ALL share/schema/test_ai_column_schema.cpp)
target_include_directories(ai_column_schema_tests PRIVATE "${DEP_DIR}/include")
target_link_libraries(ai_column_schema_tests PRIVATE ob_base oceanbase
  "${DEP_DIR}/lib/libgtest_main.a" "${DEP_DIR}/lib/libgtest.a" pthread)
add_test(NAME ai_column_schema_tests COMMAND ai_column_schema_tests)
