cmake_minimum_required(VERSION 2.8)

project(sphere_problem)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++17")

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# Eigen3
include_directories("/usr/include/eigen3")

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


########################################################################

add_executable(ceresSphereQuatAuto pose_graph_ceres_quaternion_sphere.cpp)
target_link_libraries(ceresSphereQuatAuto ${CERES_LIBRARIES} Sophus::Sophus)

