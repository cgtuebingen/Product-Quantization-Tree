#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cassert>

#include "helper.hpp"
#include "filehelper.hpp"

#include <gflags/gflags.h>


using namespace std;

DEFINE_string(bvecs, "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1B/bigann_query.bvecs" , "(input) path to bvec file");
DEFINE_string(umem, "/tmp/query.umem" , "(output) path to umem file");
DEFINE_int32(chunkSize, 100000 , "number of vectors per chunk");

int main(int argc, char *argv[]) {
  // parse flags
  gflags::SetUsageMessage("This script converts bvecs into umem files\n"
                          "Usage:\n"
                          "    convert_bvecs --bvecs .. --umem ... \n");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);


  const uint chunkSize = FLAGS_chunkSize;

  string fs = FLAGS_bvecs;

  // ****************************** HEADER **********************

  uint n1 = 0;
  uint d1 = 0;
  readJegouHeader<uint8_t>(fs.c_str(), n1, d1);

  cout << "header dim " << d1 << endl;
  cout << "header num " << n1 << endl;
  cout << "filesize   " << (sizeof(uint8_t)*n1 * d1) / 100000.0 << " mb " << endl;

  const uint fileSize = sizeof(uint8_t) * n1 * d1;

  n1 = 50000000;

  std::fstream fin(FLAGS_umem, std::ios_base::out | std::ios_base::binary);
  fin << n1 << std::endl;
  fin << d1 << std::endl;
  fin.ignore();
  fin.seekg( 20 , std::ios::beg );


  for (uint pos = 0; pos < n1; pos += chunkSize) {
    const uint start = pos;
    const uint end = (start + chunkSize > n1) ? n1 : start + chunkSize;
    const uint length = end - start ;

    cout << "read from " << pos << " to " << end << endl;
    uint8_t *data = readBatchJegou(fs.c_str(), start, length);
    fin.write((char*) data, length * d1 * sizeof(uint8_t));
    fin.flush();
    delete[] data;
  }
  fin.close();



  return 0;
}