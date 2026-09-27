cmake_minimum_required(VERSION 3.22)
project(dense_map)

set(CMAKE_CXX_STANDARD 17)

find_package(glim REQUIRED)

add_library(dense_map SHARED
  src/glim_ext/dense_map.cpp
)
target_link_libraries(dense_map
  glim::glim
  glim_ext
)
