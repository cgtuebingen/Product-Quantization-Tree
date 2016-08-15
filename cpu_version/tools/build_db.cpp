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
  
    treequantizer<T, D, C1, C2, P, H1, RE> Q;
    Q.loadTree("tree.tree");

    // insert db
    std::string path = "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/base.umem";
    memiterator<float, uint8_t> base_set;
    base_set.open(path.c_str());
    Q.notify(base_set.num());
    timer<> sw;
    sw.start();
    for (uint n = 0, n_e = base_set.num(); n < n_e; ++n) {
        float *lean_data  = base_set.addr(n);
        Eigen::Matrix<T, D, 1> curVec = Eigen::Map<Eigen::Matrix<T, D, 1>>(lean_data);
        Q.insert(curVec);
        delete[] lean_data;
    }
    sw.stop();
    std::cout << "time to insert db           " << sw.elapsed() << "ms" << std::endl;


    Q.saveBins("tree.bins");

    const uint countBaseVecs  = base_set.num();


    std::cout << "lambda range      " << Q.minLambdaValue  << " to " << Q.maxLambdaValue << std::endl;
    std::cout << "avg. quant error  " << Q.avgQuantError / static_cast<float>(countBaseVecs) << std::endl;
    std::cout << "min. quant error  " << Q.minQuantError << std::endl;
    std::cout << "max. quant error  " << Q.maxQuantError << std::endl;





    return 0;
}
