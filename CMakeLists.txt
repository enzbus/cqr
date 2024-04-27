cmake_minimum_required(VERSION 3.10.1)
project(project_euromir)
set(SRC
  project_euromir/linear_algebra.c
)
add_library(project_euromir SHARED ${SRC})
