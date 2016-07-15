#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cassert>

#include "../helper.hpp"
#include "../filehelper.hpp"
#include "../iterator/bvecsiterator.hpp"

using namespace std;

int main(int argc, char const *argv[]) {

    string umem = "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1B/base.umem";
    string bvecs = "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1B/bigann_base.bvecs";

    std::ifstream _handle(umem.c_str(), std::ios_base::in | std::ios_base::binary);
    if (!_handle.good()) {
        _handle.close();
        throw std::runtime_error("read error for file " + umem);
    }

    uint nn = 0;
    uint dd = 0;
    
    _handle >> nn;
    _handle >> dd;
    _handle.ignore();
    _handle.seekg( 0, std::ios::beg );
    _handle.seekg( 20, std::ios::beg );

 
    const uint chunksize = 1000000;
    uint8_t *db = new uint8_t[128 * chunksize];
    float *conv_data = new float[128 * chunksize];

    for (uint vecCounter = 0, n_e = 1000000000; vecCounter < n_e; vecCounter += chunksize) {
        _handle.read((char*) db, 128 * chunksize * sizeof(uint8_t));
        uint8_t *data2 = readBatchJegou(bvecs.c_str(), vecCounter, chunksize);

        for (int i = 0; i < chunksize*128; ++i)
        {
            if(db[i] != data2[i])
                cout << "missmatch at "<< i << endl;
        }


        delete[] data2;

    }



    return 0;
}