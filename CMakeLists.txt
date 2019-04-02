# Author: Emilie Gillet (emilie.o.gillet@gmail.com)

cmake_minimum_required(VERSION 2.6)

enable_testing()

project(beta_nmf)

add_definitions(
  -Wall -O2 -DNDEBUG -D_FILE_OFFSET_BITS=64 -fPIC -Wno-sign-compare -msse2
)

include_directories(
    .
    src
    third_party
    third_party/gtest/
    third_party/gtest/include
    third_party/eigen3/
)

link_directories(${LINK_DIRECTORIES})

set(EXECUTABLE_OUTPUT_PATH build)
set(LIBRARY_OUTPUT_PATH build)
set(CMAKE_CURRENT_SOURCE_DIR .)

# add_library(
#   beta_nmf
# )

add_library(
  gtest
    third_party/gtest/src/gtest-all.cc
)

# target_link_libraries(
#   beta_nmf
# )

macro(register_test TEST_NAME)
  add_executable(${TEST_NAME} tests/${TEST_NAME}.cpp)
  target_link_libraries(
    ${TEST_NAME}
      # beta_nmf
      gtest)
  add_test(tests/${TEST_NAME} ${EXECUTABLE_OUTPUT_PATH}/${TEST_NAME})
endmacro(register_test)

file(GLOB tests_files tests/*.cpp)
foreach(test_file ${tests_files})
  get_filename_component(test_file_name ${test_file} NAME_WE)
  register_test(${test_file_name})
  message("Registering test: " ${test_file_name})
endforeach()
