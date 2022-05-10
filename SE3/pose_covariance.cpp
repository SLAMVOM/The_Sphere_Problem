/***************************************************************************
* This file contains the definitions of the covariance calculation functions
* that are used in the Sphere problem.
*
* The covariance calculation follows the approach stated in A.3.1 in
* Prof. Tim Barfoot's book: State Estimation for Robotics
*
* Author: MT
* Creation Date: 2022-May-10
* Previous Edit: 2022-May-10
***************************************************************************/

#include "pose_covariance.h"

void calPoseCovariance(const int N_vertices,                      // input
                       const Eigen::SparseMatrix<double> &A_mat,  // input
                       CovMap &vertex_covariances                 // output
                      )
{
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    int n_rows = A_mat.rows(); // get the number of rows of the A_mat
    int n_cols = A_mat.cols(); // get the number of cols of the A_mat

    if (n_rows != n_cols) {
        std::cout << "The A matrix should be a symmtric matrix, but the input has "
                  << n_rows << " rows and "
                  << n_cols << " columns." << std::endl;
        exit(1);
    }
    if (n_rows % N_vertices != 0) {
        std::cout << "The dimension of the input A matrix is not divisible by the number of vertices. Check matrix dimension."
                  << std::endl;
    }

    // Define an Eigen sparse cholesky solver
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseMatrix<double> L_mat = solver.compute(A_mat).matrixL(); // obtain the lower block triangular matrix

    // In this case, the covariance matrix for each pose is a 6x6 matrix
    // using a for loop to compute the covariance associated with each pose
    for (int i = N_vertices-1; i >= 0; i--) { // starting from
        Eigen::SparseMatrix<double> L_k_s = L_mat.block(i*6, i*6, 6, 6); // (startRow, startCol, rows, cols)
        Eigen::MatrixXd L_k_d;
        L_k_d = Eigen::MatrixXd(L_k_s); // convert the sparse matrix to a dense matrix
        Matrix6d L_k_inv_d = L_k_d.inverse(); // inversion of a small dense matrix would not be so expensive

        if (i < N_vertices-1) { // if not the last covariance block
            // when it is not the last cov block, we need the block below the diagonal block
            Eigen::SparseMatrix<double> L_k_k_1_s = L_mat.block(i*6+6, i*6, 6, 6); // L_k,k-1, the block below L_k
            Eigen::MatrixXd L_k_k_1_d;
            L_k_k_1_d = Eigen::MatrixXd(L_k_k_1_s); // convert the sparse matrix to a dense matrix
            auto P_k_plus1 = vertex_covariances.at(i+1);
            Matrix6d P_k =   L_k_inv_d.transpose()
                          * (Matrix6d::Identity()
                                + L_k_k_1_d.transpose() * P_k_plus1 * L_k_k_1_d)
                          *  L_k_inv_d;

            if (vertex_covariances.find(i) == vertex_covariances.end()) { // if the key is not already in the map
                vertex_covariances.insert(std::make_pair<int, Matrix6d>(i*1, P_k*1.0));
            } else { // if the key is already in the map
                vertex_covariances[i] = P_k;
            }
        } else { // the last covariance matrix block only relates to the bottom right block in the L_mat
            Matrix6d P_K = L_k_inv_d.transpose() * L_k_inv_d;
            if (vertex_covariances.find(i) == vertex_covariances.end()) { // if the key is not already in the map
                vertex_covariances.insert(std::make_pair<int, Matrix6d>(i*1, P_K*1.0));
            } else { // if the key is already in the map
                vertex_covariances[i] = P_K;
            }
        }
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    std::cout << "Time spent for calculating "<< N_vertices
              << " covariance matrices: " << time_used.count() << "sec." << std::endl;

} // end of calPoseCovariance()
