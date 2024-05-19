cmake_minimum_required(VERSION 3.15)
project(project_euromir)

set(SRC
  project_euromir/linear_algebra.c
)

set(CMAKE_BUILD_TYPE "Release")

add_library(project_euromir SHARED ${SRC})
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# stay away from visual studio, would require bunch of hacks in the code
if(WIN32)
  set(CMAKE_GENERATOR "MinGW Makefiles")
endif()

set(CMAKE_C_FLAGS "-O3 -Wall -fbounds-check -Wextra")
set(CMAKE_VERBOSE_MAKEFILE 1)

install(TARGETS project_euromir
    DESTINATION ${CMAKE_SOURCE_DIR}/project_euromir
    COMPONENT project_euromir)