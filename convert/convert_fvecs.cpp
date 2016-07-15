#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cassert>

#include "helper.hpp"
#include "filehelper.hpp"

#include <gflags/gflags.h>

using namespace std;

typedef float out_t;

DEFINE_string(fvecs, "/graphics/projects/scratch/ANNSIFTDB/ANN_RAND100K/base.fvecs" , "(input) path to fvec file");
DEFINE_string(umem, "/tmp/base.umem" , "(output) path to umem file");

int main(int argc, char *argv[]) {
  // parse flags
  gflags::SetUsageMessage("This script converts ivecs into imem files\n"
                          "Usage:\n"
                          "    convert_fvecs --fvecs .. --umem ...  \n");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  string path = FLAGS_fvecs;
  // ****************************** HEADER **********************

  uint n1 = 0;
  uint d1 = 0;
  readJegouHeader<float>(path.c_str(), n1, d1);

  cout << "header dim " << d1 << endl;
  cout << "header num " << n1 << endl;
  cout << "filesize   " << (sizeof(float)*n1 * d1) / 1000000.0 << " mb " << endl;
  // ****************************** READ **********************


  uint n = 0;
  uint d = 0;

  float *data = readJegou<float>(path.c_str(), n, d);

  cout << "load dim " << d << endl;
  cout << "load num " << n << endl;

  cout << "dump:" << endl;
  for (int j = 0; j < 5; ++j) {
    cout << j << ": ";
    for (int i = 0; i < 10; ++i) {
      cout << data[j * d + i] << " ";
    }
    cout << " ..." << endl;
  }
  cout << endl;

  // ****************************** CONVERT **********************
  out_t *out = batchcast<float, out_t>(data, d * n);

  // ****************************** WRITE **********************
  write(FLAGS_umem, n, d, out, n * d);

  // ****************************** TEST **********************
  uint nn = 0;
  uint dd = 0;
  out_t *test_in = new out_t[n * d];
  read(FLAGS_umem, nn, dd, test_in, n * d);

  cout << "test dump:" << endl;
  cout << "load dim " << dd << endl;
  cout << "load num " << nn << endl;
  for (int j = 0; j < 5; ++j) {
    cout << j << ": ";
    for (int i = 0; i < 10; ++i) {
      cout << (float)test_in[j * d + i] << " ";
    }
    cout << " ..." << endl;
  }
  cout << endl;

  for (int j = 0; j < nn; ++j) {

    for (int i = 0; i < dd; ++i) {
      if (test_in[j * d + i] != out[j * d + i]) {
        cout << "ERROR: MISSMATCH vector " << j << " coordinate " << i << endl;
        return 1;
      }
    }
  }

  cout << "INFO: identical files" << endl;

  return 0;
}