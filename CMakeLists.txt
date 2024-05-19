cmake_minimum_required(VERSION 3.10)
project(project_euromir)

set(SRC
  project_euromir/linear_algebra.c
)

add_library(project_euromir SHARED ${SRC})

set(CMAKE_C_FLAGS "-O3 -Wall -fbounds-check -Wextra")
set(CMAKE_VERBOSE_MAKEFILE 1)

install(TARGETS project_euromir
    DESTINATION ${CMAKE_SOURCE_DIR}/project_euromir
    COMPONENT project_euromir)