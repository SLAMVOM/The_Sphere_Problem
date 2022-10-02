/****************************************************************************
* This program demonstrates how to use g2o solver for pose graph optimization
* sphere.g2o is a synthetic pose graph.
* Although the whole graph can be loaded by calling the `load` function,
* we still implement the data loading to obtain a better understanding of the data.
* This script uses the SE3 pose in g2o/types/slam3d/,
* which represents the rotation using quaternion rather than Lie Algebra
****************************************************************************/

#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

using namespace std;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage pose_graph_g2o_SE3 sphere.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    // setup g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // graph model
    optimizer.setAlgorithm(solver);     // setting solver
    optimizer.setVerbose(true);         // turn on verbose output

    int vertexCnt = 0, edgeCnt = 0;     // counting number of vertices and edges
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // SE3 vertex
            g2o::VertexSE3 *v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            if (index == 0)
                v->setFixed(true);      // fixing the first vertex (not changing)
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 edge
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1, idx2;             // the indices of the correspondences
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    cout << "saving optimization results ..." << endl;
    optimizer.save("result.g2o");

    return 0;
}
