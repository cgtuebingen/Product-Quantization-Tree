#include <iostream>
#include "pqt/bitonicSort.cuh"
#include "pqt/triangle.cuh"
#include "utils/helper.hpp"
#include <stdio.h>
#include <stdlib.h>
using namespace pqt;

void testSortLarge() {

  dim3 block(256, 1, 1);
  dim3 grid(1, 1, 1);

  sortTestLarge<<<grid, block, 8 * 1024>>>(1024);
  sortTestLarge<<<grid, block, 8 * 2048>>>(2048);
  sortTestLarge<<<grid, block, 8 * 4096>>>(4096);

  SAFE_CUDA_CALL(cudaDeviceSynchronize());
  std::cout << "done SortTest " << std::endl;

  block = dim3(1024, 1, 1);
  scanTestLarge<<<grid, block, 4 * 1024>>>(1024);
  scanTestLarge<<<grid, block, 4 * 2048>>>(2048);
  scanTestLarge<<<grid, block, 4 * 4096>>>(4096);

  SAFE_CUDA_CALL(cudaDeviceSynchronize());

  std::cout << "done Scan Test" << std::endl;
  std::cout.flush();

}

void testDistances() {

  float a2, b2, c2, d2, lambda;
  float dd2;
  bool test = true;

  a2 = 1.f;
  b2 = 2.f;
  c2 = 1.f;
  lambda = project(a2, b2, c2, d2);
  dd2 = dist(a2, b2, c2, lambda);

  std::cout << "(1,1) ==? " << lambda << " " << d2 << "  == ? dist: " << dd2
      << std::endl;

  test = test && equal(1.f, lambda) && equal(1.f, d2) && equal(d2, dd2);

  a2 = 2.f;
  b2 = 2.f;
  c2 = 4.f;
  lambda = project(a2, b2, c2, d2);
  dd2 = dist(a2, b2, c2, lambda);

  std::cout << "(0.5,1) ==? " << lambda << " " << d2 << "  == ? dist: " << dd2
      << std::endl;
  test = test && equal(0.5f, lambda) && equal(1.f, d2) && equal(d2, dd2);

  a2 = 2.f;
  b2 = 2.f;
  c2 = 2.f;
  lambda = project(a2, b2, c2, d2);
  dd2 = dist(a2, b2, c2, lambda);

  std::cout << "(0.5,1.5) ==? " << lambda << " " << d2 << "  == ? dist: " << dd2
      << std::endl;
  test = test && equal(.5f, lambda) && equal(1.5f, d2) && equal(d2, dd2);

  a2 = 2.f;
  b2 = (4.f + 1.f);
  c2 = 9.f;
  lambda = project(a2, b2, c2, d2);
  dd2 = dist(a2, b2, c2, lambda);

  std::cout << "(.666, 1) ==? " << lambda << " " << d2 << "  == ? dist: " << dd2
      << std::endl;
  test = test && equal(0.666666666f, lambda) && equal(1.f, d2)
      && equal(d2, dd2);

  a2 = 2.f;
  b2 = (4.f + 1.f);
  c2 = 1.f;
  lambda = project(a2, b2, c2, d2);
  dd2 = dist(a2, b2, c2, lambda);

  std::cout << "(2, 1) ==? " << lambda << " " << d2 << "  == ? dist: " << dd2
      << std::endl;
  test = test && equal(2.f, lambda) && equal(1.f, d2) && equal(d2, dd2);

  b2 = 2.f;
  a2 = 4.f + 1.f;
  c2 = 1.f;
  lambda = project(a2, b2, c2, d2);
  dd2 = dist(a2, b2, c2, lambda);

  std::cout << "(-1, 1) ==? " << lambda << " " << d2 << "  == ? dist: " << dd2
      << std::endl;
  test = test && equal(-1.f, lambda) && equal(1.f, d2) && equal(d2, dd2);

  if (test)
    std::cout << "distance test passed!" << std::endl;
  else
    std::cout << "distance test failed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

  for (int i = -100; i < 100; i++) {

    float f = i / 10.f;

    std::cout << "\t\t" << f << "=" << toFloat(toUShort(f)) << " = "
        << toUShort(f);
  }
  std::cout << std::endl;
  //exit(0);
}

int main (int argc, char **argv)
{
  testSortLarge();
  testDistances();
  return 0;
}