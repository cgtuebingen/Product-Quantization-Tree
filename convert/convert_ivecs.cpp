#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cassert>


#include "helper.hpp"
#include "filehelper.hpp"

#include <gflags/gflags.h>

using namespace std;

DEFINE_string(ivecs, "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1B/idx_50M.ivecs" , "(input) path to ivec file");
DEFINE_string(imem, "/tmp/groundtruth.imem" , "(output) path to imem file");


int main(int argc, char* argv[]) {
  // parse flags
  gflags::SetUsageMessage("This script converts ivecs into imem files\n"
                          "Usage:\n"
                          "    convert_ivecs --ivecs .. --umem ...  \n");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  string path = FLAGS_ivecs;
  // ****************************** HEADER **********************

  uint n1 = 0;
  uint d1 = 0;
  readJegouHeader<int>(path.c_str(), n1, d1);

  cout << "header dim " << d1 << endl;
  cout << "header num " << n1 << endl;
  cout << "filesize   " << (sizeof(int)*n1 * d1) / 1000000.0 << " mb " << endl;
  // ****************************** READ **********************


  uint n = 0;
  uint d = 0;

  int *data = readJegou<int>(path.c_str(), n, d);

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
  //uint8_t *out = batchcast<float,uint8_t>(data,d*n);
  int *out = data;
  // ****************************** WRITE **********************
  //write<int>( path+".imem", n, d, out, n*d);
  write<int>( FLAGS_imem,  n, d, out, n * d);

  // ****************************** TEST **********************
  uint nn = 0;
  uint dd = 0;
  int *test_in = new int[n * d];
  read( FLAGS_imem, nn, dd, test_in, n * d);

  cout << "test dump:" << endl;
  cout << "load dim " << dd << endl;
  cout << "load num " << nn << endl;
  for (int j = 0; j < 5; ++j) {
    cout << j << ": ";
    for (int i = 0; i < 10; ++i) {
      cout << (int)test_in[j * d + i] << " ";
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
  gflags::ShutDownCommandLineFlags();
  return 0;
}