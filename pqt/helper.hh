#ifndef NEARESTNEIGHBOR_HELPER_H
#define NEARESTNEIGHBOR_HELPER_H

/*! \file  helper.hh
    \brief a collection of helper classes
 */
//#define OUTPUT

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <iostream>

using namespace std;

#define MAX_THREADS 512
#define MAX_BLOCKS 65535
#define WARP_SIZE 32

namespace pqt {

inline __device__ float sqr(const float &x) {
	return x * x;
}

// returns the ceiling of the log base 2 of an integer, i.e. the mimimum number of bits needed to store x
inline unsigned int log2(unsigned int x) {
	unsigned int y;

	for (y = 0; y < 64; y++)
		if (!((x - 1) >> y))
			break;

	y = 1 << y;

	return y;
}


inline uint idiv(uint _n, uint _d) {
	uint val = _n / _d;
	return (_n % _d) ? val + 1 : val;
}


void outputMat(const std::string& _S, const float* _A,
		uint _rows, uint _cols);

void outputVec(const std::string& _S, const float* _v,
		uint _n);


void outputVecUint(const std::string& _S, const uint* _v,
		uint _n);

void outputVecInt(const std::string& _S, const int* _v,
		uint _n);

inline void setReductionBlocks(dim3& _block, uint _n) {
	unsigned int nThreads = log2(_n);
	nThreads = (nThreads < WARP_SIZE) ? WARP_SIZE : nThreads;
	nThreads = (nThreads > MAX_THREADS) ? MAX_THREADS : nThreads;
	_block = dim3(nThreads, 1, 1);
}

//__device__ void bitonic(volatile float _val[], volatile uint _idx[], uint _N);


void countZeros(const std::string& _S, const uint* _v, uint _n);





} /* namespace */



#endif /* NEARESTNEIGHBOR_HELPER_H */
