cmake_minimum_required(VERSION 2.8)

project(the_sphere_problem)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++14")

# Eigen3
include_directories("/usr/include/eigen3")

add_library(covariance_calculation pose_covariance.cpp)

#add_executable(testing test.cpp)

add_executable(poseGraphGN pose_graph_GN.cpp)

add_executable(poseGraphCovGN pose_graph_GN_cov.cpp)
target_link_libraries(poseGraphCovGN covariance_calculation)
