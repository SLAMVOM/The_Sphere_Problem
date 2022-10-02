/*****************************************************************
* This code is to implmenet the BALL problem in Ch10 of VSLAM book
* The pose graph optimization is carried out in Ceres solver,
* with unit quaternion and auto diff.
*
* References:
*  - Section 10.3 in VSLAM book 2nd ed.
*  - https://github.com/ceres-solver/ceres-solver/blob/master/examples/slam/pose_graph_3d/pose_graph_3d_error_term.h
*
* The solver used in this problem is Ceres.
*
* Created by: MT
* Creation Date: 2022-May-01
* Previous Edit: 2022-May-02
*****************************************************************/

#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>

using namespace std;

// define some types to be used
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

struct Pose3d {
    Eigen::Vector3d p;
    Eigen::Quaterniond q;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}; // Pose3d

// From Ceres' example:
// "Computes the error term for two poses that have a relative pose measurement
// between them. Let the hat variables be the measurement. We have two poses x_a
// and x_b. Through sensor measurements we can measure the transformation of
// frame B w.r.t frame A denoted as t_ab_hat. We can compute an error metric
// between the current estimate of the poses and the measurement.
//
// In this formulation, we have chosen to represent the rigid transformation as
// a Hamiltonian quaternion, q, and position (or translation), p. The quaternion ordering is
// [x, y, z, w].
//
//
// Note: R(q_a) is R_{ia}, where i denotes the inertial world frame, thus,
// R(q_a)^T is R_{ai} is the rotation from the inertial frame to the a frame.
// p_a is p_i^{ai}, indicating the translation from the inertial frame to the a frame
// expressed in the inertial frame.
// Therefore, p_b - p_a = p_i^{bi} - p_i^{ai} = p_i^{ba} is the translation vector
// from the a frame to the b frame expressed in the inertial frame.
// R(q_a)^T * (p_b - p_a) = R_{ai} * p_i^{ba} = p_a^{ba} is the translation vector
// from the a frame to the b frame expressed in the a frame.
// And, R(q_a^{-1] * q_b) = R_{ab} is the rotation from the b frame to the a frame (or called
// the rotation of frame B w.r.t. frame A).
// The transformation from the frame B to the frame A: T_{ab}
// T_{ab} = [R_{ab} | t_a^{ba}]
//          [0  0  0|        1]
//
// In vector format:
// The estimated measurement (7D vector: 3D translation + 4D unit quaternion) is:
//      t_ab = [ p_ab ]  = [ R(q_a)^T * (p_b - p_a) ]
//             [ q_ab ]    [ q_a^{-1] * q_b         ]
//
// where ^{-1} denotes the inverse and R(q) is the rotation matrix for the
// quaternion. Now we can compute an error metric between the estimated and
// measurement transformation. For the orientation error, we will use the
// standard multiplicative error resulting in:
//
//   error = [ p_ab - \hat{p}_ab                 ]
//           [ 2.0 * Vec(q_ab * \hat{q}_ab^{-1}) ]
//
// where Vec(*) returns the vector (imaginary) part of the quaternion. Since
// the measurement has an uncertainty associated with how accurate it is, we
// will weight the errors by the square root of the measurement information
// matrix:
//
//   residuals = I^{1/2) * error
// where I is the information matrix which is the inverse of the covariance."
class PoseGraph3dErrorTerm {
    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PoseGraph3dErrorTerm(Pose3d t_ab_measured, Eigen::Matrix<double, 6, 6> sqrt_information)
            : t_ab_measured_(std::move(t_ab_measured)),
              sqrt_information_(std::move(sqrt_information)) {}

        template <typename T>
        bool operator()(const T* const pos_a_ptr,
                        const T* const qua_a_ptr,
                        const T* const pos_b_ptr,
                        const T* const qua_b_ptr,
                        T* residuals_ptr) const {
            // Note when using Eigen::Map to cast a double array into an Eigen::Quaternion,
            // the qw should be located at the end!!!
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos_a(pos_a_ptr); // cast the position vec of A into a Eigen Vec
            Eigen::Map<const Eigen::Quaternion<T>> qua_a(qua_a_ptr); // cast the rotation vec of A into a Eigen Quaternion

            Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos_b(pos_b_ptr); // cast the position vec of B into a Eigen Vec
            Eigen::Map<const Eigen::Quaternion<T>> qua_b(qua_b_ptr); // cast the rotation vec of B into a Eigen Quaternion

            // Compute the relative transformation from frame B to frame A
            Eigen::Quaternion<T> qua_a_inverse = qua_a.conjugate(); // vector form of R_{ai}, i stands for inertial
            Eigen::Quaternion<T> qua_ab_estimated = qua_a_inverse * qua_b; // vector form of R_{ab}

            // Represent the displacement between the two frames in the A frame
            // i.e., the translation from frame A to frame B expressed in frame A, t_a^{ba}
            Eigen::Matrix<T, 3, 1> trans_ab_estimated = qua_a_inverse * (pos_b - pos_a);

            // Compute the error between the two orientation estimates in frame A => d_qua = R_{ab, meas} * R_{ba, esti}, can be thought of as estimated (-) measured
            Eigen::Quaternion<T> delta_qua = t_ab_measured_.q.template cast<T>() * qua_ab_estimated.conjugate();

            // Compute the residuals.
            // [ position         ]   [ delta_pos          ]
            // [ orientation (3x1)] = [ 2 * delta_qua(0:2) ]
            Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr); // Eigen::Map wrap the array to the desired type
            residuals.template block<3, 1>(0, 0) = trans_ab_estimated - t_ab_measured_.p.template cast<T>(); // estimated - measured
            residuals.template block<3, 1>(3, 0) = T(2.0) * (delta_qua.vec()); // the vector (imaginary) part of the Quaternion

            // Scale the residuals by the measurement uncertainty
            residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

            return true;
        }

    static ceres::CostFunction* Create(const Pose3d& t_ab_measured,
                                       const Eigen::Matrix<double, 6, 6> &sqrt_information) {
        return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(
            new PoseGraph3dErrorTerm(t_ab_measured, sqrt_information));
    }

    private:
    // The measurement for the position of B relative to A in the frame A, T_{ab, measured}
    const Pose3d t_ab_measured_;
    // The (matrix) square root of the measurement information matrix
    // Note: the information matrix is a symmetric matrix, thus, the matrix square root can be found using Cholesky decomposition
    const Eigen::Matrix<double, 6, 6> sqrt_information_;

}; // PoseGraph3dErrorTerm


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: pose_graph_g2o_SE3_lie sphere.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << "does not exist." << endl;
        return 1;
    }

    ceres::Problem problem;

    /* The data file is a .g2o file, which uses unit quaternion and translation vector to describe pose
     * The VERTEX_SE3:QUAT node has the following 8 fields: ID, tx, ty, tz, qx, qy, qz, qw
     * The EDGE_SE3:QUAT node has 30 fields: ID1, ID2, tx, ty, tz, qx, qy, qz, qw, the upper right corner of the information matrix
    **/
    // Create storage variables to store the vertices and edges data
    int N_vertices = 0; // a counter to count the number of vertices in the dataset
    int N_constraints = 0; // a counter to count the number of edges (i.e., constraints) in the dataset

    std::map<int, double*> vertex_poses; // using a std::map to store the poses, <vertex_idx, pose_params>

    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> sqrt_info_matrices; // store sqrt of information matrices
    std::vector<int*> edge_vertices;// store the vertex indices related to a particular edge
    std::vector<Pose3d*> edge_T_vectors; // store the transformation Pose3d objects between the two vertex frames

    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            N_vertices++; // keep counting the number of vertices
            int index; // index of the vertex node
            fin >> index;
            double* v = new double[7]; // create a new double array to store params for each vertex

            // Note: when using Eigen::Map to cast an array into an Eigen::Quaternion object,
            // the q_w should locate at the end.
            for (int i = 0; i < 7; i++) {
                fin >> v[i];
            }
            vertex_poses.insert(std::make_pair<int, double*>(index*1, &(v[0])));

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
            double* T_vec = new double[7];
            fin >> T_vec[0] >> T_vec[1] >> T_vec[2] >> T_vec[3] >> T_vec[4] >> T_vec[5] >> T_vec[6];
            Pose3d* pose3d_vec = new Pose3d;
            pose3d_vec->p = Eigen::Vector3d(T_vec[0], T_vec[1], T_vec[2]);
            pose3d_vec->q = Eigen::Quaterniond(T_vec[6], T_vec[3], T_vec[4], T_vec[5]);

            // Normalize the quaternion to account for precision loss due to serialization
            pose3d_vec->q.normalize();
            edge_T_vectors.push_back(pose3d_vec); // store the pose3d_vec pointer

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
            // since an information matrix is symmertric, we can use the Cholesky decomposition to get the sqrt matrix
            sqrt_info_matrices.push_back(info_mat.llt().matrixL());
        }
    } // end reading file while loop

    // Building the problem
    for (int i = 0; i < N_constraints; i++) {
        ceres::CostFunction *cost_function;

        // Each residual block takes a Pose3d object and a sqrt_information matrix
        // and output a 6D residual
        cost_function = PoseGraph3dErrorTerm::Create(*(edge_T_vectors[i]), sqrt_info_matrices[i]);

        ceres::LossFunction *loss_function = nullptr;
        // loss_function = new ceres::HuberLoss(1.0); // If enabled, using Huber's loss function.

        problem.AddResidualBlock(cost_function, // cost function ptr
                                 loss_function, // robust kernel or nullptr
                                 vertex_poses[edge_vertices[i][0]], // extract value from a std::map, position of Frame A
                                 vertex_poses[edge_vertices[i][0]]+3, // the unit quaternion part of Frame A
                                 vertex_poses[edge_vertices[i][1]], // extract value from a std::map, position of Frame B
                                 vertex_poses[edge_vertices[i][1]]+3  // the unit quaternion part of Frame B
                                 );
    }
    // set the pose of the first vertex node fixed to eliminate the gauge freedom
    problem.SetParameterBlockConstant(vertex_poses[0]);
    problem.SetParameterBlockConstant(vertex_poses[0]+3);

    // show verbose information
    std::cout << "Sphere problem file loaded..." << std::endl;
    std::cout << "Sphere problem has " << N_vertices << " vertices and "
              << N_constraints << " constraints. " << std::endl;
    std::cout << "Forming " << N_constraints << " edges. " << std::endl;

    // setup the ceres solver
    // More options see: http://ceres-solver.org/nnls_solving.html
    ceres::Solver::Options options;

    //// Linear solver options
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; // specify the solver type
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // specify the solver type

    //// Minimizer options
    options.max_num_iterations = 50; // Ceres' default: 50
    options.minimizer_progress_to_stdout = true; // print out some progress information
    options.num_threads = 1; // Ceres' default: 1 - number of threads used by Ceres to evaluate the Jacobian
    options.max_solver_time_in_seconds = 1e6; // Ceres' default: 1e6 - max amount of time for the solver to run

    // Start solving the problem
    std::cout << "Solving veres BA ... " << std::endl;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::cout << "saving optimization results ..." << std::endl;
    // since the vertex was customly defined, saving the data explictly
    // pretending the data as SE3 vertex and edge to allow loading into g2o_viewer
    ofstream fout("result_ceres.g2o");
    for (auto const& pose : vertex_poses) {
        // the quaternion was casted using Eigen::Map, so the q_w is still the last element
        fout << "VERTEX_SE3:QUAT"
             << ' ' << pose.first
             << ' ' << pose.second[0]
             << ' ' << pose.second[1]
             << ' ' << pose.second[2]
             << ' ' << pose.second[3]
             << ' ' << pose.second[4]
             << ' ' << pose.second[5]
             << ' ' << pose.second[6]
             << '\n';
    }
    for (int i = 0; i < N_constraints; i++) {
        fout << "EDGE_SE3:QUAT"
             << ' ' << edge_vertices[i][0]
             << ' ' << edge_vertices[i][1]
             << ' ' << edge_T_vectors[i]->p[0]
             << ' ' << edge_T_vectors[i]->p[1]
             << ' ' << edge_T_vectors[i]->p[2]
             << ' ' << edge_T_vectors[i]->q.x()
             << ' ' << edge_T_vectors[i]->q.y()
             << ' ' << edge_T_vectors[i]->q.z()
             << ' ' << edge_T_vectors[i]->q.w();

        for (int j = 0; j < 6; j++) {
            for (int k = j; k < 6; k++) {
                fout << ' ' << (sqrt_info_matrices[i]*sqrt_info_matrices[i])(j,k);
            }
        }
        fout << '\n';
    }
    fout.close();

    std::cout << "Finished writing the output file." << std::endl;

    return 0;
}
