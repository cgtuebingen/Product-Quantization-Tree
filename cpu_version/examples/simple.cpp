#include <iostream>
#include <string>
#include "../helper.hpp"
#include "../iterator/memiterator.hpp"
#include "../iterator/iterator.hpp"
#include "../quantizer/treequantizer.hpp"

const uint P = 2;     // number of parts
const uint C1 = 16;   // number of clusters in 1st level per part
const uint C2 = 8;    // number of refinements
const uint H1 = 4;
const uint RE = 32;

typedef Eigen::Matrix < float, 128 , 1 > vec_t;

int main(int argc, char const *argv[])
{

    // the data structure
    treequantizer<float, 128, C1, C2, P, H1, RE> tree;

    // build tree
    float *trainingData = new float[100000 * 128];
    iterator<float, 128> trainingIter;
    trainingIter.insertBatch(trainingData, uint 100000);
    tree.generate(trainingIter);


    // build db
    tree.notify(100);
    float *vec = new float[128];
    Eigen::Matrix<float, 128, 1> curVec = Eigen::Map<Eigen::Matrix<float, 128, 1>>(vec);
    tree.insert(curVec);

    // query
    std::vector<std::pair<uint, float>> vectorCandidates;    // proposed vectors (ordered)

    float *search = new float[128];
    Eigen::Matrix<float, 128, 1> searchVec = Eigen::Map<Eigen::Matrix<float, 128, 1>>(search);
    tree.query(20000, 500, searchVec,  vectorCandidates);

    std::cout << " best vector id " << vectorCandidates[0].first;
    std::cout << " best vector distance " << vectorCandidates[0].second;


    return 0;
}