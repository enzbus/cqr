cmake_minimum_required(VERSION 3.10)
project(project_euromir)

set(SRC
  project_euromir/linear_algebra.c
  project_euromir/dcsrch.c
  project_euromir/dcstep.c
  # trying to plug in lbfgsb
  project_euromir/Lbfgsb.3.0/blas.f
  project_euromir/Lbfgsb.3.0/linpack.f
  project_euromir/Lbfgsb.3.0/lbfgsb.f
  project_euromir/Lbfgsb.3.0/timer.f
)

enable_language(Fortran)

SET(
  MOVE_SHLIB_TO ${CMAKE_SOURCE_DIR}/project_euromir
  CACHE FILEPATH "Filepath of installation."
)
message("THE SHARED LIBRARY WILL BE COPIED INTO = ${MOVE_SHLIB_TO}")

add_library(project_euromir SHARED ${SRC})

set(CMAKE_C_FLAGS "-O3 -Wall -fbounds-check -Wextra")

# copied from Lbfgsb Makefile
set(CMAKE_Fortran_FLAGS "-O -Wall -fbounds-check -g -Wno-uninitialized")

set(CMAKE_VERBOSE_MAKEFILE 1)

# flags for f2c, temporary
target_link_libraries(
  project_euromir
  #PUBLIC f2c
  PUBLIC m)

install(TARGETS project_euromir
    #DESTINATION ${CMAKE_SOURCE_DIR}/project_euromir
    #DESTINATION ${CMAKE_INSTALL_PREFIX}
    DESTINATION ${MOVE_SHLIB_TO}
    COMPONENT project_euromir)