#include <iostream>
#include <string>
#include "../helper.hpp"
#include "../timer.hpp"
#include "../iterator/memiterator.hpp"
#include "../iterator/iterator.hpp"
#include "../quantizer/treequantizer.hpp"


const uint D = 128;   // dimension of vector
const uint P = 2;     // number of parts
const uint C1 = 16;   // number of clusters in 1st level per part
const uint C2 = 8;    // number of refinements
const uint H1 = 4;
const uint RE = 32;
typedef float T;




int main(int argc, char const *argv[]) {
    UNUSED(argc);
    UNUSED(argv);

    std::string path = "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/offline.umem";

    // load learn set for codebook creation
    memiterator<float, uint8_t> learn_set;
    learn_set.open(path.c_str());
    std::cout << "dim:                        " << learn_set.dim() << std::endl;
    std::cout << "num:                        " << learn_set.num() << std::endl;
    iterator<float, 128> iter_learn;
    float *lean_data = learn_set.all();
    iter_learn.insertBatch(lean_data, learn_set.num());

    // generate structure
    treequantizer<T, D, C1, C2, P, H1, RE> Q;
    Q.generate(iter_learn);
    Q.saveTree("tree.tree");


    return 0;
}
