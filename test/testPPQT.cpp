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

void analyze(vector<uint>& _resIdx, uint _nvec, int* _GT, uint _gt_dim,
		uint _QN) {

	float foundBest = 0;

	float percentTop10 = 0.;
	float percentTop100 = 0;

	float r10 = 0.;
	float r100 = 0.;
	float rTotal = 0.;

	for (int v = 0; v < _QN; v++) {

		if (_resIdx[v * _nvec] == _GT[v * _gt_dim]) {
			cout << _resIdx[v * _nvec] << " gt: " << _GT[v * _gt_dim] << endl;
			foundBest += 1.;
		}

		if (_nvec > 10) {
			float p = 0.;
			for (uint i = 0; i < 10; i++) {

				uint idx = _GT[v * _gt_dim + i];

				bool found = false;
				for (int k = 0; k < _nvec; k++) {
					if (_resIdx[v * _nvec + k] == idx) {
						found = true;
						break;
					}
				}

				if (found)
					p += 1. / 10.;
			}
			percentTop10 += p;

			uint idx0 = _GT[v * _gt_dim];
			for (int k = 0; k < 10; k++) {
				if (_resIdx[v * _nvec + k] == idx0) {
					r10 += 1.;
					break;
				}
			}
		}

		if (_nvec > 100) {
			float p = 0.;
			for (uint i = 0; i < 100; i++) {

				uint idx = _GT[v * _gt_dim + i];

				bool found = false;
				for (int k = 0; k < _nvec; k++) {
					if (_resIdx[v * _nvec + k] == idx) {
						found = true;
						break;
					}
				}

				if (found)
					p += 1. / 100.;
			}
			percentTop100 += p;

			uint idx0 = _GT[v * _gt_dim];
			for (int k = 0; k < 100; k++) {
				if (_resIdx[v * _nvec + k] == idx0) {
					r100 += 1.;
					break;
				}
			}
		}

		uint idx0 = _GT[v * _gt_dim];
				for (int k = 0; k < _nvec; k++) {
					if (_resIdx[v * _nvec + k] == idx0) {
						rTotal += 1.;
						break;
					}
				}
	}

	foundBest /= _QN;
	percentTop10 /= _QN;
	percentTop100 /= _QN;

	r10 /= _QN;
	r100 /= _QN;
	rTotal /= _QN;

	cout << setprecision(4) << _nvec << "\t" << foundBest << "\t"
			<< percentTop10 << "\t" << percentTop100 << endl;
	cout << "R10 / R100 / R" << _nvec << ": " << r10 << "\t" << r100 << "\t" << rTotal << endl;
}


int bestGpuDevice(){
  int ret = system("../../tools/bestgpu/gpu.pyc");
  return WEXITSTATUS(ret);
}

int main(int argc, char* argv[]) {


	int bGPU = bestGpuDevice();

	cout << "starting on GPU " << bGPU << endl;

	cudaSetDevice( bGPU );

//	cudaSetDevice(0);
//
//	ProTree ptt(128, 4, 4);
//	ptt.testScan();
//
//	return 1;
//

	if (argc < 2) {
		cout << "usage: testVQ <baseName> <dim> <p> <nCluster1> <nCluster2> "
				<< endl;
	}

	int acount = 1;

	string uniformName = "NN-data/cb";
	string baseName = string(argv[acount++]);

	uint dim;
	uint N;
	uint QN;
	int p;
	uint nCluster1;
	uint nCluster2;

	dim = atoi(argv[acount++]);
	p = atoi(argv[acount++]);
	nCluster1 = atoi(argv[acount++]);
	nCluster2 = atoi(argv[acount++]);

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
	cudaMalloc(&Qd, QN * dim * sizeof(float));
//	cudaMalloc(&Distd, N * QN * sizeof(float));
	cudaMalloc(&closestd, QN * sizeof(int));

	cudaMemcpy(Md, M, N * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Qd, Q, QN * dim * sizeof(float), cudaMemcpyHostToDevice);

	int k = 16;

	PerturbationProTree ppt(dim, p, p, 1);
//	PerturbationProTree ppt(dim, p, p, 2);
//	PerturbationProTree ppt(dim, p, p, 3);

	ppt.testSortLarge();

	ppt.testDistances();

	string cbName = baseName + "_" + intToString(dim) + "_" + intToString(p)
			+ "_" + intToString(nCluster1) + "_" + intToString(nCluster2) + "_"
			+ intToString(ppt.getNPerturbations()) + ".ppqt";

	cout << "trying to read" << cbName << endl;
	if (!file_exists(cbName)) {
//		pt.createTree(nCluster1, nCluster2, Md, 100000);
		ppt.createTree(nCluster1, nCluster2, Md, 300000);
		//	pt.testCodeBook();
		ppt.writeTreeToFile(cbName);
	} else {
		ppt.readTreeFromFile(cbName);
	}

//	ppt.buildDB(Md, N);

	ppt.buildKBestDB(Md, N);
//	ppt.buildKBestDB(Md, N/10);

	ppt.lineDist(Md, N);


//
//	ppt.buildKBestLineDB(Md, N);

#if 0
	/******* test brute force comparison to 256^2 cluster centers *******/

	cout << "starting brute force comparison test" << endl;
	uint* assignd;
	cudaMalloc(&assignd, 4 * 1000 * 20 * sizeof(uint));

	for (int k = 4; k <= 16 ; k++) {
	uint nComparisons = pow(2,k);
	cout << "nComparisons: " << nComparisons << endl;
//	getKBestAssignment(assignd, d_multiCodeBook, _Q, d_nClusters, _QN, 1);
	ppt.getAssignment(assignd, Md, Qd, nComparisons, 10000);
	cout << "." << endl;
	}

	cudaFree(assignd);

	cout << "ended brute force comparison test" << endl;

	/* end * test brute force comparison to 256^2 cluster centers *******/


	exit(0);
#endif
	cudaFree(Md);


#if 0
//	ppt.testKNN(Qd + 3 * dim, 1);
	ppt.testKNN(Qd, 1);

#else
	vector<uint> resIdx;
	vector<float> resDist;

//	QN = 1000;

//	for (int i =0 ; i < 101; i++)

	uint vecs[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
//	for (int i = 13; i < 14; i++) {
//		for (int i = 9; i < 10; i++) {
	{
//		uint nVec = vecs[i];
		uint nVec = 4096;
		ppt.queryKNN(resIdx, resDist, Qd, QN, nVec);

		analyze(resIdx, nVec, GT, gt_dim, QN);
	}

#endif
	cudaFree(closestd);
//	cudaFree(Distd);
	cudaFree(Qd);
//	cudaFree(Md);

//	delete[] closest;
//	delete[] D;

	delete[] Q;
	delete[] M;

	cout << endl << "done" << endl;

	cout.flush();

}

#endif /* NEARESTNEIGHBOR_TESTVQ_C */
