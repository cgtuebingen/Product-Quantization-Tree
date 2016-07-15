#ifndef NEARESTNEIGHBOR_TESTVQ_C
#define NEARESTNEIGHBOR_TESTVQ_C

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <cstring>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "VectorQuantization.hh"
#include "ProductQuantization.hh"
#include "ProQuantization.hh"
#include "ProTree.hh"
#include "PerturbationProTree.hh"

#include "readSIFT.hh"

#include <sys/stat.h>
#include <sstream>
#include <iomanip>

#include <algorithm>

using namespace std;
// using namespace base;
// using namespace image;

using namespace nearestNeighbor;
using namespace nearestNeighborPQ;

bool file_exists(const std::string& _name) {
	struct stat buffer;

	return (stat(_name.c_str(), &buffer) == 0);
}

string intToString(const uint _x) {

	stringstream sstr;

	sstr << _x;
	return sstr.str();
}


int bestGpuDevice(){
  int ret = system("../../tools/bestgpu/gpu.pyc");
  return WEXITSTATUS(ret);
}

int main(int argc, char* argv[]) {


	int bGPU = bestGpuDevice();

	cout << "starting on GPU " << bGPU << endl;

	cudaSetDevice( bGPU );


	if (argc > 2) {
		cout << "usage: testBrute "
				<< endl;
	}

	int acount = 1;


	string uniformName = "NN-data/cb";
	string baseName = "bla"; // string(argv[acount++]);

	uint dim;
	uint N;
	uint QN;
	int p;
	uint nCluster1;
	uint nCluster2;

	dim = 128;
	p = 1;
	nCluster1 = 16;
	nCluster2 = 16;

	float *M;
	float *Q;

	int* GT;
	size_t gt_dim = 0, gt_num = 0;

	if (baseName == uniformName) {

		N = 1000000;
		QN = 1000;

		// generate random vectors
		M = new float[N * dim];
		for (int i = 0; i < N * dim; i++) {
			M[i] = drand48() - 0.5;

		}

		for (int i = 0; i < dim; i++) {
			cout << M[i] << " ";
		}

		Q = new float[QN * dim];
		for (int i = 0; i < QN * dim; i++) {
			Q[i] = drand48() - 0.5;
		}
	} else {

		const char *path_learn =
				"/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/learn.fvecs";
		const char *path_query =
				"/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/query.fvecs";
		const char *path_base =
				"/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/base.fvecs";
		const char *path_truth =
				"/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/groundtruth.ivecs";

		/*
		 1 Million vecs
		 dimension   nb base vectors   nb query vectors  nb learn vectors
		 128          1,000,000           10,000           100,000
		 */

		size_t query_dim = 0, base_dim = 0;
		size_t query_num = 0, base_num = 0;

		//	 float *learn_db = fromFile<float>(path_learn, learn_dim, learn_num);
		Q = fromFile<float>(path_query, query_dim, query_num);
		M = fromFile<float>(path_base, base_dim, base_num);

		GT = fromFile<int>(path_truth, gt_dim, gt_num);

		cout << "GT  ";
		for (int i = 0; i < gt_dim; i++) {
			cout << "\t" << GT[3 * gt_dim + i];
		}
		cout << endl;

		cout << "GT dim: " << gt_dim << "  GT_num: " << gt_num;
		QN = query_num;
		N = base_num;
		if (base_dim != dim)
			dim = base_dim;

	}

	cout << "created vectors" << endl;

	float *Md, *Qd;
	float *Distd;
	int *closestd;

	cudaMalloc(&Md, N * dim * sizeof(float));

	QN = 128;

	cout << "N: " << N << " QN: " << QN << endl;

	cudaMalloc(&Qd, QN * dim * sizeof(float));
	cudaMalloc(&Distd, N * QN * sizeof(float));

	cudaMemcpy(Md, M, N * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Qd, Q, QN * dim * sizeof(float), cudaMemcpyHostToDevice);

	int k = 16;

	PerturbationProTree ppt(dim, p, p, 1);

	vector<float> dVec;
		dVec.resize(QN * N);
		float *d =  &dVec[0];

	cout << "starting dist calc" << endl;

	ppt.calcDist(Distd,Md, Qd, N, QN, dim, 1);

#if 0

	cudaMemcpy(d, Distd, QN * N * sizeof(float), cudaMemcpyDeviceToHost);

	// sort each vector independently
	cout << "start sorting " << endl;
	for (int i = 0; i < QN; i++) {

		std::sort(dVec.begin() + (i * N), dVec.begin() + ((i+1) * N));
	}
#else
	cout << "start sorting " << endl;

	for (int i = 0; i < QN; i++) {

		ppt.parallelSort(Distd, (i*N), (i+1)*N);

	}

	cudaMemcpy(d, Distd, QN * N * sizeof(float), cudaMemcpyDeviceToHost);
#endif

	cout << "done" << endl;
	for (int i = 0; i < QN; i++) {
		cout << d[i * N] << " - " << d[i*N + 1] << "    ";
	}
	cout << endl;

	cudaFree(Md);
	cudaFree(Distd);
	cudaFree(Qd);


	delete[] Q;
	delete[] M;

	cout << endl << "done" << endl;

	cout.flush();

}

#endif /* NEARESTNEIGHBOR_TESTVQ_C */
