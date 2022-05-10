/*****************************************************************
* This code is to implmenet the sphere.g2o problem by hand.
* The optimization algorithm used: Gauss-Newton.
*
* Reference: Prof. Tim Barfoot's Textbook Section 8.3
*
* Created by: MT
* Creation Date: 2022-May-04
* Previous Edit: 2022-May-10
*****************************************************************/

/* Data format of the sphere.g2o file:
* The first part of the data starts with the tag "Vertex_SE3:QUAT",
* indicating a node/vertex of the pose graph. The second part of the file
* includes data entries start with "EDGE_SE3:QUAT", which indicates an edge
* within the graph that connects two poses. The g2o file uses quaternion
* and 3D translation vector to represent poses.
*
* Each VERTEX_SE3:QUAT node has the following eight fields: ID, tx, ty, tz, qx, qy, qz, qw.
* After the node ID, the first three are translation vector elements, while the last four
* are unit quaternion elements that represent rotation.
* The transformation matrix can be thought of as a transformation from the current frame with an index specified by the 'ID' entry
* to the  inertial frame, Frame 0 (assumed to be Identity).
* One may also interpret the transformation as:
* a transformation matrix representing the pose of the stationary inertial frame (i.e., Frame 0) relative to the ID's Frame.
* That is the given elements recover a matrix of: T_{0a}, where a is ID; and 0 indicates the inertial frame.
* Thus, T_{0a} = [R_{0a} | t_0^{a0}]
*                [0  0  0|        1]
*
* For the EDGE_SE3:QUAT rows, each row has 30 fields: ID1, ID2, tx, ty, tz, qx, qy, qz, qw, the upper right corner of the information matrix.
* The first two ID's are the indicies of the "to" and "from" vertices, respectively.
* That is, the transformation matrix can transfrom a point from Frame ID2 to Frame ID1.
* One may also interpret the matrix as: a transformation matrix representing pose of Frame ID1 relative to Frame ID2.
* The remaining 21 elements are the diagonal and upper right part of an information matrix related to the current relative transformation.
* Since the information matrix is a symmetric matrix,
* providing half of the matrix and the diagonal will be enough to recover the whole matrix.
* Therefore, the transformation matrix is from Frame ID2 to Frame ID1,
* or called the pose of Frame ID2 w.r.t. Frame ID1: T_{ab}
* Thus, T_{ab} = [R_{ab} | t_a^{ba}]
*                [0  0  0|        1]
*
*/

/* The workflow of the program:
* 1. Reading the vertex and edge data from the file
* 2. Calculate and constructing the error term at the operating point, Sigma matrix, G matrix, and P matrix associated with each edge
* 3. Constructing the [6K x 6K] A matrix and [6K x 1] b vector for calculating the [6K x 1] delta_x vector
* 4. Extracting each [6 x 1] update vector to update the pose through exponential map
* 5. Iterate from 2 to 4 until a stopping criterion is met
*/


#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <typeinfo> // get type of a variable, for debugging

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>

#include "lie_utils.h"
#include "pose_covariance.h"

#define CAL_COV     1   // a flag to set if covariance should be calculated, any int > 0 means to calculate

using namespace std;

// define some types to be used
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 4, 4> Matrix4d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;


int main(int argc, char **argv) {

    auto n_thd = Eigen::nbThreads();
    std::cout << n_thd << " number of thread(s) will be used in Eigen." << std::endl;

    if (argc != 2) {
        cout << "Usage: pose_graph_g2o_SE3_lie sphere.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << "does not exist." << endl;
        return 1;
    }

    //// Define some user-specified variables for the problem
    int num_max_iter = 25; // the maximum number of iterations for the optimization
    double delta_thresh = 1e-4; // threshold to determine if the change of variables is small enough to claim a convergence
    int converge_flag = 0; // a flag to denote if convergence has been reached, > 0 denotes convergence


    //// Step 1: Reading the vertex and edge data from the file
    // Create storage variables to store the vertices and edges data
    int N_vertices = 0; // a counter to count the number of vertices in the dataset
    int N_constraints = 0; // a counter to count the number of edges (i.e., constraints) in the dataset

    // if compliling in prior to [c++17], needs to allocate as follows
    // Note that the third parameter std::less<int> is just the default value, it has to be included to allow specifying the allocator type
    std::map<int, Matrix4d, std::less<int>,
             Eigen::aligned_allocator<std::pair<const int, Matrix4d>>> vertex_poses; // using a std::map to store the poses, <vertex_idx, pose_params>

    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> info_matrices; // store the information matrices
    std::vector<int*> edge_vertices;// store the vertex indices related to a particular edge
    std::vector<Matrix4d, Eigen::aligned_allocator<Matrix4d>> edge_T_matrices; // store the transformation matrices between two vertex frames
    //if the CAL_COV flag is greater than 0, then calculating covariance associated with each pose
    std::map<int, Matrix6d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Matrix6d>>> vertex_covariances;

    // Read the data file into appropriate data variables
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            N_vertices++; // keep counting the number of vertices
            int index; // index of the vertex node
            fin >> index;
            double v[7]; // create a new double array to store params for each vertex

            // Note: when using Eigen::Map to cast an array into an Eigen::Quaternion object,
            // the q_w should locate at the end.
            for (int i = 0; i < 7; i++) {
                fin >> v[i];
            }

            // Here, converting the translation vector and unit quaternion to a SE(3) Transformation matrix
            // when representing a unit quaternion the real part w should come the first
            Vector4d tmp_qua(v[6], v[3], v[4], v[5]); // a temporary vector representing the unit quaternion

            double ang_axis[3]; // angle-axis representation of the same rotatoin as the unit quaternion
            unitQuaternionToAngleAxis(tmp_qua.data(), ang_axis); // convert unit quaternion to angle-axis vector

            double phi_hat_arr[9]; // this array stores the row-major skew-symmetric matrix for the rotation vector
            vecHat(ang_axis, phi_hat_arr); // convert the angle-axis vector to a skew-symmetric matrix stored in an 1d array

            Eigen::Matrix3d R_mat; // this is the rotation matrix associated with the unit quaternion
            expPhiHat(Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(phi_hat_arr), R_mat); // obtain the rotation matrix

            // construct the 4 x 4 transformation matrix
            Matrix4d T_mat = Matrix4d::Zero(); // reset the matrix to zeros
            T_mat.block<3, 3>(0, 0) = R_mat; // the upper-left 3x3 block is the rotation
            T_mat.block<3, 1>(0, 3) = Eigen::Vector3d(v[0], v[1], v[2]); // the first 3 elements of the last col is translation
            T_mat(3,3) = 1.0; // set the last element to 1

            // IMPORTANT: the given translation and rotation combination from the g2o file
            //  will come up with transformation matrices of T_{0a}.
            // The derivation carried out in Barfoot's book was based on: T_{a0},
            // therefore, we need to invert the pose before processing.
            vertex_poses.insert(std::make_pair<int, Matrix4d>(index*1, T_mat.inverse()*1.0)); // inserting to std::map for storage

        } else if (name == "EDGE_SE3:QUAT") {
            // SE3 - SE3 edge
            N_constraints++; // keep counting the number of edges

            // Extracting and storing the indices of the two associated vertices
            int* indices_arr = new int[2];
            fin >> indices_arr[0] >> indices_arr[1]; // indices of a pair of correspondences
            edge_vertices.push_back(indices_arr);

            // Extracting and storing the relative pose between the two vertices, T_ab, from frame B to frame A
            // According to Eigen, when constructing a quaternion by the Quaternion() method,
            // the real part w should come the first,
            // while internally the coefficients are stored in the order of [x,y,z,w]
            double T_vec[7];
            fin >> T_vec[0] >> T_vec[1] >> T_vec[2] >> T_vec[3] >> T_vec[4] >> T_vec[5] >> T_vec[6];

            // Here, converting the translation vector and unit quaternion to a SE(3) Transformation matrix
            // when representing a unit quaternion the real part w should come the first
            Vector4d tmp_qua(T_vec[6], T_vec[3], T_vec[4], T_vec[5]); // a temporary vector representing the unit quaternion

            double ang_axis[3]; // angle-axis representation of the same rotatoin as the unit quaternion
            unitQuaternionToAngleAxis(tmp_qua.data(), ang_axis); // convert unit quaternion to angle-axis vector

            double phi_hat_arr[9]; // this array stores the row-major skew-symmetric matrix for the rotation vector
            vecHat(ang_axis, phi_hat_arr); // convert the angle-axis vector to a skew-symmetric matrix stored in an 1d array

            Eigen::Matrix3d R_mat; // this is the rotation matrix associated with the unit quaternion
            expPhiHat(Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(phi_hat_arr), R_mat); // obtain the rotation matrix

            // construct the 4 x 4 transformation matrix
            Matrix4d T_mat = Matrix4d::Zero(); // reset the matrix to zeros
            T_mat.block<3, 3>(0, 0) = R_mat; // the upper-left 3x3 block is the rotation
            T_mat.block<3, 1>(0, 3) = Eigen::Vector3d(T_vec[0], T_vec[1], T_vec[2]); // the first 3 elements of the last col is translation
            T_mat(3,3) = 1.0; // set the last element to 1

            edge_T_matrices.push_back(T_mat); // store the transformation matrix

            // construct the information matrix
            Matrix6d info_mat = Matrix6d::Zero();
            for (int i = 0; i < info_mat.rows(); i++) {
                for (int j = i; j < info_mat.cols(); j++) {
                    fin >> info_mat(i, j);
                    if (i != j) {
                        info_mat(j, i) = info_mat(i, j);
                    }
                }
            }
            // store the information matrices
            info_matrices.push_back(info_mat);
        }
    } // end reading file while loop

    std::cout << "Finished reading the data file into storage variables." << std::endl;

    // First layer for loop to perform the optimization iteratively
    for (int iter_idx = 0; iter_idx < num_max_iter; iter_idx++){
        std::cout << "\n--------- Iteration: " << iter_idx+1 << " ---------" << std::endl;

        //// Step 2. Calculate and constructing the error term at the operating point, Sigma matrix, G matrix, (and P matrix) associated with each edge
        //// Step 3. Constructing the [6K x 6K] A matrix and [6K x 1] b vector for calculating the [6K x 1] delta_x vector
        // first define the A matrix and b vector => A * dx = b
        // Note: we can treat the elements in A matrix and b vector as parameter blocks.
        // In A, each block is a 6x6 matrix, while in b, each block is 6x1.
        // Eigen::MatrixXd A_mat(6*N_vertices, 6*N_vertices); // [6K x 6K] matrix, NOTE: constructing A_mat in advance may possibly lead to out of memory issue
        Eigen::MatrixXd b_vec(6*N_vertices, 1); // [6K x 1] residual vector


        // Initialize the matrix and vector as zeros
        // A_mat.setZero(); // NOTE: constructing A_mat in advance may possibly lead to out of memory issue when Method 1 (see below) is used
        b_vec.setZero();

        // Instead of using the projection matrix, P_kl as in the derivation, construct a big G matrix
        Eigen::MatrixXd Big_G(6*N_constraints, 6*N_vertices); // this is similar to the Jacobian matrix in Bundle Adjustment
        Big_G.setZero();

        // Define a variable to record the accumulated translational residual
        double err_sq_trans = 0.0;

        // second layer for loop to go through all the edges to construct the problem
        for (int i = 0; i < N_constraints; i++) {
            // Here we use subscript k, l to denote the ID1 frame and ID2 frame in each constraint, respectively.

            // Inverse Sigma matrix, using the information matrix in this case - a symmetric matrix stored in info_matrices
            // In this dataset, it is not only a block diagonal matrix, but also a diagonal matrix
            Matrix6d Sigma_kl_inv = info_matrices[i]; // [6 x 6]

            // e_{kl}(x_{op}) - the error term for the k-l edge at the current operating point
            // each error term is a 6 x 1 vector
            // e_{kl}(x_{op}) = ln( T_{kl, meas} * T_{l0,op} * T_{k0,op}^{-1} )^{v}
            Matrix4d T_kl_meas, T_k_op, T_l_op, T_delta;
            T_kl_meas = edge_T_matrices[i]; // [4 x 4]
            T_k_op = vertex_poses[edge_vertices[i][0]]; // [4 x 4]
            T_l_op = vertex_poses[edge_vertices[i][1]]; // [4 x 4]
            T_delta = T_kl_meas * T_l_op * T_k_op.inverse(); // [4 x 4], note the error will be in the ID1 frame
            Vector6d e_kl_op; // define the error term vector [6 x 1]
            lnVeeToZeta(T_delta, e_kl_op); // the err term is the lie algebra zeta vec associated with the transformation mat

            // accumulate the squared translational error
            err_sq_trans += T_delta.block<3,1>(0,3).transpose() * T_delta.block<3,1>(0,3);

            // G matrix - the first 6x6 block is w.r.t. the ID2 frame, and the second 6x6 block is w.r.t. the ID1 frame
            // each G matrix is a 6 x 12 matrix
            // G = [ -J_left(-e_{kl}(x_{op}))^(-1) * Adj(T_{k0,op}) * Adj(T_{l0,op})^(-1)  |  J_left(-e_{kl}(x_{op}))^(-1) ]
            Eigen::Matrix<double, 6, 12> G_mat; // [6 x 12]
            Matrix6d Inv_left_Jac_e_kl_op, Adj_T_k, Adj_T_l_inv; // some matrices to be used for calculating G
            // First block term w.r.t. ID2
            invLeftJacobianSE3(-e_kl_op, Inv_left_Jac_e_kl_op); // inverted left Jacobian mat w.r.t. -e_{kl}(x_{op}), [6 x 6]
            AdjointSE3(T_k_op, Adj_T_k); // the Adjoint matrix associated with T_{k0,op}, [6 x 6]
            invAdjointSE3(T_l_op, Adj_T_l_inv); // the inverted Adjoint matrix associated with T_{l0, op}, [6 x 6]
            G_mat.block<6, 6>(0, 0) = - Inv_left_Jac_e_kl_op * Adj_T_k * Adj_T_l_inv; // [6 x 6]
            // Second block term w.r.t. ID1
            G_mat.block<6, 6>(0, 6) = Inv_left_Jac_e_kl_op;



            ///////////////// Method 1 : Using Big_G matrix and perform Big_G^T * Big_G to obtain A_mat /////////////////
            // in the b_kl term, the first 6 elements are w.r.t. ID2, the second half is w.r.t. ID1
            Eigen::Matrix<double, 12, 1> b_kl = G_mat.transpose() * Sigma_kl_inv * e_kl_op; // [12 x 1]
            b_vec.block<6, 1>(edge_vertices[i][1]*6, 0) += b_kl.block<6,1>(0, 0); // accumulates terms associated with ID2
            b_vec.block<6, 1>(edge_vertices[i][0]*6, 0) += b_kl.block<6,1>(6, 0); // accumulates terms associated with ID1

            // Since the information matrix is symmetric in this case, we can obtain matrix square root by Chelesky decomposition
            Matrix6d sqrt_info_mat = Sigma_kl_inv.llt().matrixL(); // this is the matrix square root of the information matrix, [6 x 6]
            Eigen::Matrix<double, 6, 12> Weighted_G_mat = sqrt_info_mat * G_mat; // [6 x 12]
            Big_G.block<6, 6>(i*6, edge_vertices[i][1]*6) = Weighted_G_mat.block<6, 6>(0, 0); // first 6 cols - ID2
            Big_G.block<6, 6>(i*6, edge_vertices[i][0]*6) = Weighted_G_mat.block<6, 6>(0, 6);// second 6 cols - ID1



            ///////////////// Method 2: Using projection matrix and accumulate A_mat and b_vec /////////////////
            /*
            ** In theory we can follow the steps as the derivation in Barfoot's book, including the useage of
            ** Projection matrices, P_kl. However, the size of the matrix is large and majority of the data
            ** entries are zeros in the matrix, resulting in very expensive and redudant computation. Most of the
            ** additions are 0 + 0, which are not particularly useful for obtaining the results. Therefore,
            ** it would be more appropriate to update the corresponding block in A_mat and b_vec directly without
            ** using the P matrix. So the below block of code will run slowly and cannot scale.
            */

            // // P matrix - first row of block corresponds to the ID2 and second for ID1 in each constraint,
            // // each P matrix is a 12 x 6K matrix
            // Eigen::MatrixXd P_kl(12, 6*N_vertices); // [12 x 6K]      //// CHECK FOR ORDERING OF THE BLOCKS
            // P_kl.block<6, 6>(0, 6*edge_vertices[i][1]) = Matrix6d::Identity(); // Frame ID2 block as identity
            // P_kl.block<6, 6>(6, 6*edge_vertices[i][0]) = Matrix6d::Identity(); // Frame ID1 block as identity

            // Accumulate the A_mat and b_vec
            // A_mat += P_kl.transpose() * G_mat.transpose() * Sigma_kl_inv * G_mat * P_kl;
            // b_vec += P_kl.transpose() * G_mat.transpose() * Sigma_kl_inv * e_kl_op;

        }

        std::cout << "The total squared translation error is: " << err_sq_trans << std::endl;
        std::cout << "Start solving for dx" << std::endl;


        //////////////// If using Method 1 stated above and have enough RAM, uncomment the following lines ////////////////
        // Eigen::MatrixXd A_mat = Big_G.transpose() * Big_G; // note: the square root information matrix is embedded in the Big_G matrix

        // //// Using regular Eigen Solvers
        // // Solve for the optimal perturbation: A * dx = b
        // // Note:: Matrix A is a symmetric matrix, so that we can use cholesky decomposition to solve
        // auto dx = A_mat.ldlt().solve(b_vec); // pick one of the linear solver methods
        // auto dx = A_mat.llt().solve(b_vec); // pick one of the linear solver methods
        // auto dx = A_mat.partialPivLu().solve(b_vec); // pick one of the linear solver methods



        //////////////// using Method 1 above but with limited RAM, then use sparse matrices; otherwise comment the follows ////////////////
        Eigen::SparseMatrix<double> Big_G_sparse = Big_G.sparseView();
        Big_G.resize(0,0); // freeing/destructing the memory on Big_G matrix, otherwise, may subject to out of memory issue
        auto A_mat = Big_G_sparse.transpose() * Big_G_sparse;

        // //// Using sparse Eigen Solvers
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver; // pick one of the sparse solvers
        // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver; // pick one of the sparse solvers
        solver.compute(A_mat);
        auto dx = solver.solve(b_vec);
        if (solver.info() != Eigen::Success) {
            std::cout << "Failed sovling for dx." << std::endl;
        }


        std::cout << "Finished solving for dx, start updating" << std::endl;
        // 4. Extracting each [6 x 1] update vector to update the pose through exponential map
        for (auto &pose : vertex_poses) {
            if (pose.first == 0) {
                std::cout << "The transformation matrix of the first pose is not updated" << std::endl;
                continue;
            }

            // // pose updates are carried out as left multiplication
            Matrix4d T_update;
            zetaToSE3(dx.block<6, 1>(pose.first*6, 0), T_update);
            pose.second = T_update * pose.second;
        }


        //// calculate covariance matrix related to each pose if CAL_COV is greater than 0
        if(CAL_COV > 0) {
            calPoseCovariance(N_vertices, A_mat, vertex_covariances); // using a std::map to store the ID-Cov_mat pairs
            // to show several calculated covariance matrices
            for (int l = 2; l < 4; l++) {
                std::cout << "\nCovariance of pose " << l << ":\n" << vertex_covariances[l] << std::endl;
            }
        }



        // //// Check if any convergence threshold is met, if converged, break the optimization loop

        if (dx.array().abs().maxCoeff() < delta_thresh) { // L-inf norm is smaller than the threshold
            std::cout << "The L-Inf norm of the update vector is smaller than threshold " <<  delta_thresh
                      << "-> Converged." << std::endl;
            converge_flag += 1;
        }


        if (converge_flag > 0) {
            std::cout << converge_flag << " convergence condition(s) is/are met. Breaking the optimization loop." << std::endl;
            break;
        }

    } // end of first layer optimization for loop

    if (converge_flag == 0) { // no convergence is claimed
        std::cout << "\nNo convergence condition was met after " << num_max_iter
                  << " iterations (i.e., num_max_iter specified by the user is reached)." << std::endl;
    }

    std::cout << "\n====================\nFinished optimization.\n" << std::endl;

    std::cout << "Saving the optimized poses to file." << std::endl;
    // since the vertex was customly defined, saving the data explictly
    // pretending the data as SE3 vertex and edge to allow loading into g2o_viewer
    ofstream fout("result_GN.g2o"); // open up the output file, create one if not already exists
    for (auto const& pose : vertex_poses) {
        // IMPORTANT: the given translation and rotation combination from the g2o file is for
        // transformation matrix of T_{0a}.
        // The derivation carried out in Barfoot's book was based on: T_{a0},
        // and the matrices stored in vertex_poses were in the form of T{a0}.
        // So we need to convert back to T_{0a} form before saving into file, i.e., perform an inversion.
        Matrix4d T_0a = pose.second.inverse(); // this follows the g2o convention

        Eigen::Matrix3d R_pose = T_0a.block<3, 3>(0, 0); // extract the rotation matrix from the pose
        double R_arr[9]; // a vector to store a raveled form of the ratation matrix - ROW MAJOR
        // Note: In Eigen, matrices are stored in column major order, but here needs ROW MAJOR, so needs to tranpose R before mapping
        Eigen::Map<Eigen::MatrixXd>(R_arr, R_pose.cols(), R_pose.rows()) = R_pose.transpose(); // map the rotation into a 1D array with 9 elements
        double unit_quaternion_arr[4]; // the unit quaternion is in the order of [qw, qx, qy, qz], qw is the scalar part
        rotationMatrixArrayToUnitQuaternion(R_arr, unit_quaternion_arr); // convert the rotation matrix array into a unit quaternion array

        // the quaternion was casted using Eigen::Map, so the q_w is still the last element
        fout << "VERTEX_SE3:QUAT"
             << ' ' << pose.first              // ID
             << ' ' << T_0a.coeff(0, 3) // tx
             << ' ' << T_0a.coeff(1, 3) // ty
             << ' ' << T_0a.coeff(2, 3) // tz
             << ' ' << unit_quaternion_arr[1]  // qx
             << ' ' << unit_quaternion_arr[2]  // qy
             << ' ' << unit_quaternion_arr[3]  // qz
             << ' ' << unit_quaternion_arr[0]  // qw
             << '\n';
    }
    for (int i = 0; i < N_constraints; i++) {
        Eigen::Matrix3d R_pose = edge_T_matrices[i].block<3,3>(0,0); // extract the rotation matrix from the transformation matrix
        double R_arr[9]; // a vector to store a raveled form of the ratation matrix - ROW MAJOR
        // Note: In Eigen, matrices are stored in column major order, but here needs ROW MAJOR, so needs to tranpose R before mapping
        Eigen::Map<Eigen::MatrixXd>(R_arr, R_pose.cols(), R_pose.rows()) = R_pose.transpose(); // map the rotation into a 1D array with 9 elements
        double unit_quaternion_arr[4]; // the unit quaternion is in the order of [qw, qx, qy, qz], qw is the scalar part
        rotationMatrixArrayToUnitQuaternion(R_arr, unit_quaternion_arr); // convert the rotation matrix array into a unit quaternion array

        fout << "EDGE_SE3:QUAT"
             << ' ' << edge_vertices[i][0]           // ID1
             << ' ' << edge_vertices[i][1]           // ID2
             << ' ' << edge_T_matrices[i].coeff(0,3) // tx
             << ' ' << edge_T_matrices[i].coeff(1,3) // ty
             << ' ' << edge_T_matrices[i].coeff(2,3) // tz
             << ' ' << unit_quaternion_arr[1]        // qx
             << ' ' << unit_quaternion_arr[2]        // qy
             << ' ' << unit_quaternion_arr[3]        // qz
             << ' ' << unit_quaternion_arr[0];       // qw

        for (int j = 0; j < 6; j++) {
            for (int k = j; k < 6; k++) {
                fout << ' ' << info_matrices[i](j,k);
            }
        }
        fout << '\n';
    }
    fout.close();

    std::cout << "Finished writing the output file." << std::endl;


    return 0;
}
