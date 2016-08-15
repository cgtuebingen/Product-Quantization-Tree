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


uint ListLen = 8192;
typedef float T;

void recall(
    iterator<T, D> &iter_query,
    iterator<int, 100> &iter_truth,
    treequantizer<T, D, C1, C2, P, H1, RE> &Q,
    uint num) {
    typedef  std::pair< uint, T > bin_t;

    // r = 1,10,100,1000,10000,100000
    const int H = 6;
    uint Rs[7] = {1, 10, 100, 1000, 10000, 100000, 1000000};
    uint good[7] = {0};

    double avg_candlist_len = 0;
    double avg_binlist_len = 0;


    for (int i = 0, i_e = num; i < i_e; ++i) {
        // extract correct groundtruth id from dataset
        const uint correctId = iter_truth[i](0, 0);
        // compute requirements
        std::vector<std::pair<uint, T>> vectorCandidates;    // proposed vectors (ordered)
        Q.query(20000,500, iter_query[i],  vectorCandidates);

        // avg_binlist_len += binCandidates.size();
        avg_candlist_len += vectorCandidates.size();
        // get statistics
        bool found = false;
        uint s = 0;
        const uint s_e = min(vectorCandidates.size(),  (uint)100000);
        for (; s < s_e; ++s) {
            if (vectorCandidates[s].first == correctId) {
                found = true;
                break;
            }
        }

        if (found) {
            if (s < 1)
                ++good[0];
            if (s < 10)
                ++good[1];
            if (s < 100)
                ++good[2];
            if (s < 1000)
                ++good[3];
            if (s < 10000)
                ++good[4];
            if (s < 100000)
                ++good[5];
            if (s < 1000000)
                ++good[6];
        }


    }

    std::cout << "candidate list:             " << avg_candlist_len / static_cast<double>(num) << std::endl;
    std::cout << "bin list:                   " << avg_binlist_len / static_cast<double>(num) << std::endl;

    for (int r = 0; r < H; ++r) {
        std::cout << "@R" << Rs[r] << ": " << good[r] / static_cast<T>(num) << std::endl;
    }


}



int main(int argc, char const *argv[]) {
    UNUSED(argc);
    UNUSED(argv);

    ListLen = 20000;

    std::string path = "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/";
    std::string query = path + "query.umem";
    std::string truth = path + "groundtruth.imem";


    // load query set
    memiterator<float, uint8_t> query_set;
    query_set.open(query.c_str());
    std::cout << "query-set                   " << query << std::endl;
    std::cout << "dim:                        " << query_set.dim() << std::endl;
    std::cout << "num:                        " << query_set.num() << std::endl;
    iterator<float, 128> iter_query;
    float *query_data = query_set.all();
    iter_query.insertBatch(query_data, query_set.num());


    // load groundtruth set
    memiterator<int, int> truth_set;
    truth_set.open(truth.c_str());
    std::cout << "truth-set                   " << truth << std::endl;
    std::cout << "dim:                        " << truth_set.dim() << std::endl;
    std::cout << "num:                        " << truth_set.num() << std::endl;
    iterator<int, 100> iter_truth;
    int *truth_data = truth_set.all();
    iter_truth.insertBatch(truth_data, truth_set.num());



    treequantizer<T, D, C1, C2, P, H1, RE> Q;
    Q.loadTree("tree.tree");
    Q.loadBins("tree.bins");
    


    const uint countQueryVecs = iter_query.num();


    // query vectors
    timer<> t;
    recall( iter_query, iter_truth, Q, countQueryVecs);
    t.stop();

    std::cout << "avg. query time   " << t.elapsed() / static_cast<float>(countQueryVecs) << "ms" << std::endl;
    std::cout << "total. query time " << t.elapsed<std::chrono::seconds>()  << "s" << std::endl;




    return 0;
}
