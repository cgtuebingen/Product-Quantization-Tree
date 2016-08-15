#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cassert>

#include "../helper.hpp"
#include "../filehelper.hpp"
#include "../iterator/memiterator.hpp"
#include "../iterator/bvecsiterator.hpp"

using namespace std;

int main(int argc, char const *argv[]) {

    int line = atoi(argv[1]);

    {
        string bvecs = "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1B/bigann_base.bvecs";
        //string bvecs = "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1B/bigann_query.bvecs";
        uint8_t *data2 = readBatchJegou(bvecs.c_str(), line, 1);
        int sum = 0;
        for (int i = 0; i < 128; ++i) {
            cout << (int)data2[i] << " ";
            sum += (int)data2[i];
        }
        cout << endl;

        cout << "sum " << sum << endl;
        delete[] data2;
    }

    {
        string bvecs = "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1B/base1M.umem";
        //string bvecs = "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1B/query.umem";

        uint n = 0;
        uint d = 0;

        header(bvecs, n, d);

        cout << "header n " << n << endl;
        cout << "header d " << d << endl;


        memiterator<float, uint8_t> base_set;
        base_set.open(bvecs.c_str());
        cout << "header n " << base_set.num() << endl;

        float *lean_data  = base_set.addr(line);
        int sum = 0;
        for (int i = 0; i < 128; ++i) {
            cout << lean_data[i] << " ";
            sum += lean_data[i];
        }
        cout << endl;

        cout << "sum " << sum << endl;

        
        //read(bvecs, n, d, uint * ptr, uint len, uint offset = 0)
    }



    return 0;
}