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

int main(int argc, char* argv[]) {

//
//	ProTree ptt(128, 4, 4);
//	ptt.testScan();
//
//	return 1;
//

	if (argc < 2) {
		cout << "usage: testVQ <baseName> <dim> <p> <nCluster1> <nCluster2> " << endl;
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
	cudaMalloc(&Distd, N * QN * sizeof(float));
	cudaMalloc(&closestd, QN * sizeof(int));

	cudaMemcpy(Md, M, N * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Qd, Q, QN * dim * sizeof(float), cudaMemcpyHostToDevice);

//	VectorQuantization vq(dim);
//
//	int k = 4096;
//
//	vq.createCodeBook(k, Md, N);

//	vq.calcDist(Distd, Md, Md, N, N, dim);

//	vq.calcDist(Distd, Md, Qd, N, QN, dim);

//	ProductQuantization pq(dim, 8);

	int k = 16;

#if 0
//	pq.createCodeBook(k, Md, N);

	ProQuantization pq(dim, 8);

//	pq.testDist(Md, N);
//
	pq.testAssignment(Md, N);

	pq.testKBestAssignment(Md, N);

//
//	pq.testAvg(Md, N);

//	pq.createCodeBook(k, Md, N);
#endif

//	ProTree pt(dim, 4, 4);
//	ProTree pt(dim, nParts, nParts, 2);

	ProTree pt(dim, p, p);

	pt.testScan();

//    pt.testSort();
//
//	cout << "done intermediate !!!! " << endl;

//	 2     5     3     4
//
//	   10     9    12     8
//
//	   14    13    11    15

//	pt.createTree(16, 8, Md, N);

//	pt.createTree(32, 8, Md, N);
//
//	prepareDistSequence( pt.getClusters2() * NUM_NEIGHBORS, pt.getGroupParts() );

//	cout << "created tree" << endl;

//	pt.createTree(32, 8, Md, N);
//	pt.writeTreeToFile("tree32.pqt");

//	pt.createTree(1024, 8, Md, N);
//	pt.writeTreeToFile("tree1024.pqt");
//
//	return 0;

	string cbName = baseName + "_" + intToString(dim) + "_" + intToString(p)
			+ "_" + intToString(nCluster1) + "_" + intToString(nCluster2) + ".pqt";

	cout << "trying to read" << cbName << endl;
	if (!file_exists(cbName)) {
//		pt.createTree(nCluster1, nCluster2, Md, 100000);
		pt.createTree(nCluster1, nCluster2, Md, 300000);
	//	pt.testCodeBook();
		pt.writeTreeToFile(cbName);
	} else {
		pt.readTreeFromFile(cbName);
	}

	// pt.buildMultiDB(Md, N);
	//pt.fakeDB(Md, N);
	//pt.fakeDB( pt.getCodebook1(), pt.getNClusters() );
	// pt.fakeDB(Qd, QN);

	pt.buildDB(Md, N);

//	pt.buildDB(Md, 100000);

//	float *Dist = new float [N * QN];
//	float *closest = new float [QN];
//
//	cudaMemcpy(Dist, Distd, N * QN * sizeof(float), cudaMemcpyDeviceToHost);
//	cudaMemcpy(closest, closestD, QN * sizeof(int), cudaMemcpyDeviceToHost);

	// pt.testMultiKNN(Qd, QN);

//	pt.testMultiKNN(Md, QN);

	pt.testKNN(Qd + 3 * dim , 1);

//	pt.testLevel1(Md, QN);

//	pt.testLevel1( pt.getCodebook1(), pt.getNClusters() );
//	pt.testLevel1(Qd, QN);

#if 0

	cout << "testKNN: " << endl << endl;

	for (int vv = 0; vv < 1; vv++) {
		cout << "======================================================================" << endl;

		pt.testKNN(Md + vv * dim, QN);

		cout << "vector " << vv << endl;
		cout << "======================================================================" << endl;

	}

#endif

	cudaFree(closestd);
	cudaFree(Distd);
	cudaFree(Qd);
	cudaFree(Md);

//	delete[] closest;
//	delete[] D;

	delete[] Q;
	delete[] M;

	cout << endl << "done" << endl;

	cout.flush();

}

#endif /* NEARESTNEIGHBOR_TESTVQ_C */
