#ifndef NEARESTNEIGHBOR_TESTVQ_C
#define NEARESTNEIGHBOR_TESTVQ_C

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <cstring>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "helper.hh"

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

#if 0
uint pertIdx(uint _i, uint _dimB, uint _cb) {

	uint maxBit = _i >> _dimB;
	uint remain = _i - (maxBit << _dimB);

	return (maxBit << _cb) + ((remain >> _cb) << (_cb + 1)) + remain % (1 << _cb);

}
#endif


uint perturbIdx(uint _i, uint _dimB, uint _cb) {

	uint maxBit = _i >> _dimB;
	uint mask = (1 << _dimB) -1;
	uint remain = _i & mask;

	mask = (1 << _cb) -1;

	return (maxBit << _cb) + ((remain >> _cb) << (_cb + 1)) + (remain & mask);

}

void testPerturbation() {

	uint _dimB = 7;
	for (uint _cb = 0; _cb < 5; _cb++) {

//		for (int i = 0; i < (1 << _dimB); i++)
			for (int i = 0; i < 128; i++)
			cout << "\t" << perturbIdx(i, _dimB -1, _cb);
		cout << endl;
	}

}

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

	testPerturbation();


	if (argc < 2) {
		cout << "usage: testStatistic <baseName> <dim> <p> <maxCluster> "
				<< endl;
	}

	int acount = 1;

	string uniformName = "NN-data/cb";
	string baseName = string(argv[acount++]);

	uint dim;
	uint N;
	uint QN;
	int p;
	uint maxCluster;

	dim = atoi(argv[acount++]);
	p = atoi(argv[acount++]);
	maxCluster = atoi(argv[acount++]);

	float *M;
	float *Q;

	if (baseName == uniformName) {

//	dim = 128;
//	dim = 8;
		N = 1000000;
		QN = 1000;

		// generate random vectors
		M = new float[N * dim];
		for (int i = 0; i < N * dim; i++) {
			M[i] = drand48() - 0.5;

//		if ((i % 4) == 3)
//			M[i] = 0.;
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
				"/graphics/projects/data/ANNSIFTDB/ANN_SIFT1M/sift_learn.fvecs";
		const char *path_query =
				"/graphics/projects/data/ANNSIFTDB/ANN_SIFT1M/sift_query.fvecs";
		const char *path_base =
				"/graphics/projects/data/ANNSIFTDB/ANN_SIFT1M/sift_base.fvecs";
		const char *path_truth =
				"/graphics/projects/data/ANNSIFTDB/ANN_SIFT1M/sift_groundtruth.ivecs";

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
//	cudaMalloc(&Distd, N * QN * sizeof(float));
//	cudaMalloc(&closestd, QN * sizeof(int));

	cudaMemcpy(Md, M, N * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Qd, Q, QN * dim * sizeof(float), cudaMemcpyHostToDevice);

	vector<pair<int, float> > stats;
	stats.clear();

	vector<vector<float> > hists;
	hists.clear();
//	for (int c = 4; c < 18000; c <<= 1) {

	for (int c = 4; c <= maxCluster; c <<= 1) {
//	for (int c = 2048; c < 66000; c <<= 1) {

		cout << "number of clusters: " << c << endl;

		ProTree pt(dim, p, p, p);

		pt.prepareDistSequence(c, p);

		ProQuantization pq(dim, p);

		for (int i = 0; i < pt.getDistSeq().size(); i++) {
			cout << i << " " << pt.getDistSeq()[i].second << endl;
		}

		//	pq.testDistReal(M, Md, 10000);

		string cbName = baseName + "_" + intToString(dim) + "_" + intToString(p)
				+ "_" + intToString(c) + ".pq";

		cout << "trying to read" << cbName << endl;
		if (!file_exists(cbName)) {
			pq.createCodeBook(c, Md, 100000);
			pq.testCodeBook();
			pq.writeCodebookToFile(cbName);
		} else {
			pq.readCodebookFromFile(cbName);
		}

		pq.testKBestAssignment(Q, Qd, 100);

		vector<float> histogram;
		histogram.clear();

		float quote = pq.calcStatistics(histogram, Qd, QN, Md, 10000,
				pt.getDistSeq());

		stats.push_back(pair<int, float>(c, quote));

		hists.push_back(histogram);
	}

	cout << endl;
	cout << endl;

	cout << std::setprecision(5);

	for (int i = 0; i < stats.size(); i++)
		cout << stats[i].first << " " << stats[i].second << endl;

	for (int i = 0; i < stats.size(); i++)
		cout << "\t" << stats[i].second;
	cout << endl << endl;

	cout << "clust. \t0";
	for (int j = 1; j < hists[hists.size() - 1].size(); j <<= 1) {
		cout << "\t" << j;
	}
	cout << endl;
	for (int i = 0; i < hists.size(); i++) {
		cout << stats[i].first;
		cout << " \t" << hists[i][0];
		for (int j = 1; j < hists[i].size(); j <<= 1) {
			cout << "\t" << hists[i][j];
		}
		cout << endl;
	}

	cudaFree(Qd);
	cudaFree(Md);

	delete[] Q;
	delete[] M;

	cout << endl << "done" << endl;

	cout.flush();

}

#endif /* NEARESTNEIGHBOR_TESTVQ_C */
