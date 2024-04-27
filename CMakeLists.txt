cmake_minimum_required(VERSION 3.10.1)
project(project_euromir)
set(SRC
  project_euromir/linear_algebra.c
)
add_library(project_euromir SHARED ${SRC})
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_C_FLAGS "-O3 -Wall -Wextra -fbounds-check")
set(CMAKE_VERBOSE_MAKEFILE 1)