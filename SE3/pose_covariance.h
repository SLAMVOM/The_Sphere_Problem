/*****************************************************************************
* This file contains declarations of the covariance calculation functions
* that are used in the Sphere problem.
*
* The covariance calculation follows the approach stated in A.3.1 in
* Prof. Tim Barfoot's book: State Estimation for Robotics
*
* Note: The simpliest way to calculate the covariance matrices is to
* directly invert the A matrix in the system: A*dx=b. However, the complexity
* is O(N^3), which will be expensive if the whole problem is large,
* i.e., contains a large number of poses. Note that, we do not need all the
* covariance matrices that relate to two different poses. The only ones that we
* need are the blocks on the diagonal of the A^-1 matrix. Therefore,
* by following the method specified in Prof. Barfoot's book, we can directly
* compute the diagonal blocks, and the complexity is O(N) with a constant
* depending on the number of parameters for each pose.
*
* Author: MT
* Creation Date: 2022-May-10
* Previous Edit: 2022-May-10
*****************************************************************************/

#pragma once
#ifndef COVARIANCE_INCLUDE_H
#define COVARIANCE_INCLUDE_H

// include
#include <algorithm>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <chrono> // for timing the runnning time

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>

using namespace std;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 4, 4> Matrix4d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;


// if compliling in prior to [c++17], needs to allocate as follows
// Note that the third parameter std::less<int> is just the default value, it has to be included to allow specifying the allocator type
// using a std::map to store the vertice covariances, <vertex_idx, covariance>
typedef std::map<int, Matrix6d, std::less<int>,
                 Eigen::aligned_allocator<std::pair<const int, Matrix6d>>> CovMap;


/*
** In this function, only the covariance matrices related to each pose
** are calculated. The covariance matrix between different poses are
** not in particular interest.
** The approach and equations used in this function to compute the
** covariances are outlined in Section A.3 Prof. Barfoot's book.
**
** Input:
**      @ A_mat          : the A matrix in Ax=b, which will be used to compute the covariances
**
** Output:
**      @ covarainces_map: a std::map stores the vertex ID and corresponding covariance matrix
**                           as key-value pairs, the first element is the key indicating the ID
**                           of the vertex, the second element is the value which is a 6x6
**                           covariance matrix associated with the vertex.
**/
void calPoseCovariance(const int N_vertices,                     // input
                       const Eigen::SparseMatrix<double> &A_mat, // input
                       CovMap &vertex_covariances                // output
                      );




#endif // COVARIANCE_INCLUDE_H
