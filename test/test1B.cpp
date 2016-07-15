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

#include <stdexcept>
#include <cassert>
#include <stdint.h>

#include <stdio.h>
#include <sys/wait.h>
#include <stdlib.h>

//#define HASH_SIZE 14000000
//#define HASH_SIZE 100000000

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

void header(std::string fs, uint &num, uint &dim) {
	std::ifstream fin(fs.c_str(), std::ios_base::in | std::ios_base::binary);
	if (!fin.good()) {
		fin.close();
		throw std::runtime_error("read error");
	}
	fin >> num;
	fin >> dim;
	fin.ignore();
	fin.close();
}

template<typename T>
void read(std::string fs, T *ptr, size_t len, size_t offset = 0) {
	std::ifstream fin(fs.c_str(), std::ios_base::in | std::ios_base::binary);

	if (!fin.good()) {
		fin.close();
		throw std::runtime_error("write error");
	}

	size_t num = 0;
	size_t dim = 0;

	fin >> num;
	fin >> dim;
	fin.ignore();

	cout << "tellg: " << fin.tellg() << endl;
	cout << "offset: " << (sizeof(T) * offset) << " len: " << len << endl;

	fin.seekg(0, std::ios::beg);
	fin.seekg(20 + sizeof(T) * offset, std::ios::beg);
	fin.read((char*) ptr, (len) * sizeof(T));
	fin.close();

}

uint locate(uint _baseNum, const uint* _prefix, const uint* _counts,
		const uint* _dbIdx, uint _idx) {

	int pos;
	for (pos = 0; pos < _baseNum; pos++) {
		if (_dbIdx[pos] == _idx)
			break;
	}

	int bin;
	for (bin = 0; bin < HASH_SIZE - 1; bin++) {
		if (_prefix[bin + 1] > pos)
			break;
	}

//	cout << "idx " << _idx << " in bin: " << endl;
//	for (int i = 0; i < _counts[bin]; i++) {
//		cout << _dbIdx[_prefix[bin]+i] << " ";
//	}
//	cout << endl;

	return bin;
}

#if 0
void locateAll(uint _baseNum, const uint* _prefix, const uint* _counts,
		const uint* _dbIdx, uint _N, const vector<uint>& _gt,
		vector<uint> &_gtBins) {

	vector<uint> gtPos;
	gtPos.resize(_N);

	int pos;
	for (pos = 0; pos < _baseNum; pos++) {
		for (int i = 0; i < _N; i++) {
			if (_gt[i] == _dbIdx[pos])
			gtPos[i] = pos;
		}
	}

	cout << "located id" << endl;

	for (int i = 0; i < _N; i++) {
		_gtBins[i] = 0;
	}

	int bin;
	for (bin = 0; bin < HASH_SIZE - 1; bin++) {
		for (int i = 0; i < _N; i++) {
			if ((_gtBins[i] == 0) && (_prefix[bin + 1] > gtPos[i]))
			_gtBins[i] = bin;
		}
	}

	cout << "located bin" << endl;

}
#endif

void locateAll(uint _baseNum, const uint* _prefix, const uint* _counts,
		const uint* _dbIdx, uint _N, const vector<uint>& _gt,
		vector<uint> &_gtBins) {

	uint* binId = new uint[_baseNum];
	uint* bins = new uint[_baseNum];

	// initialize with invalid bin id
	for (int i = 0; i < _baseNum; i++)
		binId[i] = HASH_SIZE + 1;

	// fill an array that has the bin ID for each db entry
	uint pos = 0;
	for (int i = 0; i < HASH_SIZE; i++) {
		for (int k = 0; k < _counts[i]; k++, pos++)
			binId[pos] = i;
	}

	// store the binID of each db entry
	for (int i = 0; i < _baseNum; i++) {
		bins[_dbIdx[i]] = binId[i];
	}

	cout << "created id arrays" << endl;

	// select the binID for the gt vectors
	for (int i = 0; i < _N; i++) {
		_gtBins[i] = bins[_gt[i]];
	}

	cout << "located bin" << endl;

	delete[] bins;
	delete[] binId;

}

void analyze(vector<uint>& _resIdx, uint _nvec, int* _GT, uint _gt_dim,
		uint _QN, uint _baseNum = 0, const uint* _prefix = NULL,
		const uint* _counts = NULL, const uint* _dbIdx = NULL) {

	float foundBest = 0;

	float percentTop10 = 0.;
	float percentTop100 = 0;

	float r10 = 0.;
	float r100 = 0.;
	float rTotal = 0.;

	for (int v = 0; v < _QN; v++) {

		if (_resIdx[v * _nvec] == _GT[v * _gt_dim])
			foundBest += 1.;

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
		int k;
		for (k = 0; k < _nvec; k++) {
			if (_resIdx[v * _nvec + k] == idx0) {
				rTotal += 1.;
				cout << v << " idx: " << idx0 << " at loc " << k << endl;
				break;
			}
		}
#if 0
		if (k >= _nvec) {
			if (!_baseNum)
			cout << "did not find " << v << " id: " << _GT[v * _gt_dim]
			<< endl;
			else {
				cout << "did not find " << v << " id: " << _GT[v * _gt_dim];
				cout << endl;
//			cout << " bin: " << locate(_baseNum, _prefix, _counts, _dbIdx, _GT[v * _gt_dim]) << endl;
			}

		}
#endif

	}

	foundBest /= _QN;
	percentTop10 /= _QN;
	percentTop100 /= _QN;

	r10 /= _QN;
	r100 /= _QN;
	rTotal /= _QN;

	cout << setprecision(4) << _nvec << "\t" << foundBest << "\t"
			<< percentTop10 << "\t" << percentTop100 << endl;
	cout << "R10 / R100 / R" << _nvec << ": " << r10 << "\t" << r100 << "\t"
			<< rTotal << endl;
}

void analyze2(vector<uint>& _resIdx, uint _nvec, int* _GT, uint _gt_dim,
		uint _QN) {
	cout << "QN: " << _QN << " gt_dim: " << _gt_dim << endl;

	float foundBest = 0;

	float percentTop10 = 0.;
	float percentTop100 = 0;

	float r10 = 0.;
	float r100 = 0.;
	float rTotal = 0.;

	_gt_dim = 100000;

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

	cout << setprecision(4) << _nvec << "\t" << foundBest << "\t"
			<< percentTop10 << "\t" << percentTop100 << endl;
	cout << "R10 / R100 / R" << _nvec << ": " << r10 << "\t" << r100 << "\t"
			<< rTotal << endl;

	foundBest /= _QN;
	percentTop10 /= _QN;
	percentTop100 /= _QN;

	r10 /= _QN;
	r100 /= _QN;
	rTotal /= _QN;

	cout << setprecision(4) << _nvec << "\t" << foundBest << "\t"
			<< percentTop10 << "\t" << percentTop100 << endl;
	cout << "R10 / R100 / R" << _nvec << ": " << r10 << "\t" << r100 << "\t"
			<< rTotal << endl;
}

// compare on the original data base vecotrs
void analyzeM(vector<uint>& _resIdx, uint _nvec, uint _gt_dim, uint _QN,
		uint _offs) {

	cout << "QN: " << _QN << " gt_dim: " << _gt_dim << endl;

	float foundBest = 0;

	float percentTop10 = 0.;
	float percentTop100 = 0;

	float r10 = 0.;
	float r100 = 0.;
	float rTotal = 0.;

	for (int v = 0; v < _QN; v++) {

//		cout << _resIdx[v * _nvec] << endl;

		if (_resIdx[v * _nvec] == (v + _offs)) {
//			cout << _resIdx[v * _nvec] << " gt: " << v << endl;
			foundBest += 1.;
		}

		if (_nvec > 10) {

			uint idx0 = (v + _offs);
			for (int k = 0; k < 10; k++) {
				if (_resIdx[v * _nvec + k] == idx0) {
					r10 += 1.;
					break;
				}
			}
		}

		if (_nvec > 100) {

			uint idx0 = (v + _offs);
			for (int k = 0; k < 100; k++) {
				if (_resIdx[v * _nvec + k] == idx0) {
					r100 += 1.;
					break;
				}
			}
		}

		uint idx0 = (v + _offs);
		int k;
		for (k = 0; k < _nvec; k++) {
			if (_resIdx[v * _nvec + k] == idx0) {
				rTotal += 1.;
				break;
			}
		}
		if (k >= _nvec) {
			cout << "did not find match for " << v << endl;
		}
	}

	cout << setprecision(4) << _nvec << "\t" << foundBest << "\t" << endl;
	cout << "R10 / R100 / R" << _nvec << ": " << r10 << "\t" << r100 << "\t"
			<< rTotal << endl;

	foundBest /= _QN;

	r10 /= _QN;
	r100 /= _QN;
	rTotal /= _QN;

	cout << setprecision(4) << _nvec << "\t" << foundBest << "\t" << endl;
	cout << "R10 / R100 / R" << _nvec << ": " << r10 << "\t" << r100 << "\t"
			<< rTotal << endl;
}

float* readFloat(const char* _fn, size_t _dim, size_t _num, size_t _offset) {

	size_t offset = _offset * _dim;
	size_t length = _num * _dim;

	float * buf = new float[length];

	uint8_t *raw_data = new uint8_t[length];
	read<uint8_t>(_fn, raw_data, length, offset);

	for (int i = 0; i < length; i++)
		buf[i] = raw_data[i];

	delete[] raw_data;

	return buf;
}

int bestGpuDevice() {
	int ret = system("../../tools/bestgpu/gpu.pyc");
	return WEXITSTATUS(ret);
}

int main(int argc, char* argv[]) {

	int bGPU = bestGpuDevice();

	cout << "starting on GPU " << bGPU << endl;

	cudaSetDevice(1);

	// Set flag to enable zero copy access
	cudaSetDeviceFlags (cudaDeviceMapHost);

	uint mode;

//	nvmlinit();

//
//	ProTree ptt(128, 4, 4);
//	ptt.testScan();
//
//	return 1;
//

	if (argc < 2) {
		cout
				<< "usage: testVQ <baseName> <dim> <p> <nCluster1> <nCluster2> <mode> "
				<< endl;

		cout << "mode 1: generate DB" << endl;
		cout << "mode 2: evaluate precision for all number of result vectors"
				<< endl;
		cout << "mode 3: plot histogram of bins" << endl;
		cout << "mode 4: evaluate line distance reranking " << endl;
		cout << "mode 5: generate DB only for the sparsely occupied bins"
				<< endl;
		cout << "mode 6: generate DB only for the densley occupied bins"
				<< endl;
		cout << "mode 7: evaluate perfect reranking" << endl;
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

	mode = atoi(argv[acount++]);

	bool sparse = true;
	if (mode == 6)
		sparse = false;

	cout << "processing mode: " << mode << endl;

	float *M;
	float *Q;

	int* GT;
	uint gt_dim = 0, gt_num = 0;

#if 0
	const char *path_learn =
	"/scratch/lensch/ANNSIFTDB/ANNSIFTDB/ANN_SIFT1B/learn.umem";
	const char *path_query =
	"/scratch/lensch/ANNSIFTDB/ANNSIFTDB/ANN_SIFT1B/query.umem";

	const char *path_base =
	"/scratch/lensch/ANNSIFTDB/ANNSIFTDB/ANN_SIFT1B/base.umem";
	const char *path_truth =
	"/scratch/lensch/ANNSIFTDB/ANNSIFTDB/ANN_SIFT1B/groundtruth.imem";
#else
	const char *path_learn =
			"/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/learn.umem";
	const char *path_query =
			"/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/query.umem";

	const char *path_base =
			"/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/base.umem";
	const char *path_truth =
			"/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/groundtruth.imem";

#endif

	uint query_dim = 0, base_dim = 0;
	uint query_num = 0, base_num = 0;

	header(path_query, query_num, query_dim);

	//	 float *learn_db = fromFile<float>(path_learn, learn_dim, learn_num);
//	Q = fromFile<float>(path_query, query_dim, query_num);

	Q = readFloat(path_query, query_dim, query_num, 0);

	cout << "query_num: " << query_num << endl;

	// TODO !!!! here testing
	uint qOffset = 0;
//	Q = readFloat(path_base, query_dim, query_num, qOffset);

	cout << "Q[0]: ";
	for (int v = 0; v < 128; v++) {
		cout << "\t" << Q[v];
	}

	cout << "read Q " << Q[123] << endl;

	uint chunkSize = 10000000;

	header(path_base, base_num, base_dim);
	M = readFloat(path_base, base_dim, chunkSize, 0);

	cout << "read M " << M[123] << endl;

	header(path_truth, gt_num, gt_dim);
	GT = new int[gt_dim * gt_num];
//	read<int>(path_truth, GT, gt_dim, gt_num);

	read<int>(path_truth, GT, gt_dim * gt_num, 0);

	// Testing!!!!
//	gt_num = 100;

	cout << "read GT " << GT[123] << endl;

	cout << "GT  ";
	for (int i = 0; i < 100; i++) {
		//	cout << "\t" << GT[gt_dim * i];
		cout << "\t" << GT[100000 * i];
	}
	cout << endl;

	for (int i = 0; i < gt_dim * gt_num; i++)
		if (GT[i] == 81992919) {
			cout << "found number at " << i << endl;
		}
	uint gmin = GT[0];
	uint gmax = GT[0];

	for (int i = 0; i < gt_dim * gt_num; i++) {
		if (GT[i] < gmin)
			gmin = GT[i];
		if (GT[i] > gmax)
			gmax = GT[i];
	}

	cout << "GT min: " << gmin << " max: " << gmax << endl;

	cout << "GT dim: " << gt_dim << "  GT_num: " << gt_num;
	QN = query_num;

	N = chunkSize;
	if (base_dim != dim)
		dim = base_dim;

	cout << "created vectors" << endl;

	float *Md, *Qd;
	float *Distd;
	int *closestd;

	cudaMalloc(&Qd, QN * dim * sizeof(float));
	cudaMalloc(&Md, N * dim * sizeof(float));

//	cudaMalloc(&Distd, N * QN * sizeof(float));
//	cudaMalloc(&closestd, QN * sizeof(int));

	cudaMemcpy(Qd, Q, QN * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Md, M, chunkSize * dim * sizeof(float), cudaMemcpyHostToDevice);

	int k = 16;

	PerturbationProTree ppt(dim, p, p, 1);
//	PerturbationProTree ppt(dim, p, p, 2);
//	PerturbationProTree ppt(dim, p, p, 3);

	string cbName = baseName + "_" + intToString(dim) + "_" + intToString(p)
			+ "_" + intToString(nCluster1) + "_" + intToString(nCluster2) + "_"
			+ intToString(ppt.getNPerturbations()) + ".ppqt";

	cout << "trying to read" << cbName << endl;

	if (!file_exists(cbName)) {
//		pt.createTree(nCluster1, nCluster2, Md, 100000);
//		ppt.createTree(nCluster1, nCluster2, Md, 300000);

// assemble a reasonable training set
		uint nTrain = 2000000;
		float* traind;

		cudaMalloc(&traind, nTrain * dim * sizeof(float));

		if (!traind) {
			cerr << "did not get traind" << endl;
			exit(1);
		}

		for (int k = 0; k < 10; k++) {
			cudaMemcpy(traind + k * nTrain / 10 * dim,
					Md + k * chunkSize / 10 * dim,
					nTrain / 10 * dim * sizeof(float),
					cudaMemcpyDeviceToDevice);
		}

		if ((mode == 5) || (mode == 6)) {
			ppt.createTreeSplitSparse(nCluster1, nCluster2, traind, nTrain,
					sparse);
		} else
			ppt.createTree(nCluster1, nCluster2, traind, nTrain);

		cudaFree(traind);
//		ppt.createTree(nCluster1, nCluster2, Md, 7000000);
		//	pt.testCodeBook();
		ppt.writeTreeToFile(cbName);
	} else {
		ppt.readTreeFromFile(cbName);
	}

//	uint lineParts = 16;

	uint lineParts = 32;

	string lineName = baseName + "_" + intToString(dim) + "_" + intToString(p)
			+ "_" + intToString(nCluster1) + "_" + intToString(nCluster2) + "_"
			+ intToString(lineParts) + ".lines";

	string prefixName = baseName + "_" + intToString(dim) + "_" + intToString(p)
			+ "_" + intToString(nCluster1) + "_" + intToString(nCluster2)
			+ ".prefix";

	string countsName = baseName + "_" + intToString(dim) + "_" + intToString(p)
			+ "_" + intToString(nCluster1) + "_" + intToString(nCluster2)
			+ ".count";

	string dbIdxName = baseName + "_" + intToString(dim) + "_" + intToString(p)
			+ "_" + intToString(nCluster1) + "_" + intToString(nCluster2)
			+ ".dbIdx";

	uint chunkNIter = base_num / chunkSize;
//	chunkNIter = 10;
	base_num = chunkNIter * chunkSize;

	uint* binPrefix = new uint[HASH_SIZE];
	uint* binCounts = new uint[HASH_SIZE];
	uint* dbIdx = new uint[base_num];

	switch (mode) {

	case 1: {

		// building the data base
		uint* dbIdxSave = new uint[base_num];

		memset(binPrefix, 0, HASH_SIZE * sizeof(uint));
		memset(binCounts, 0, HASH_SIZE * sizeof(uint));
		memset(dbIdx, 0, base_num * sizeof(uint));

		uint* chBinPrefix = new uint[HASH_SIZE];
		uint* chBinCounts = new uint[HASH_SIZE];
		uint* chDBIdx = new uint[chunkSize];
		float* chLines = new float[chunkSize * lineParts];

		ofstream fLines(lineName.c_str(),
				std::ofstream::out | std::ofstream::binary);

//		chunkNIter = 1;

		for (int cIter = 0; cIter < chunkNIter; cIter++) {

			cout << ">>>  " << cIter << " <<<<<<<<<<<< " << endl;

			if (cIter > 0) {
				delete[] M;

				M = readFloat(path_base, base_dim, chunkSize,
						cIter * chunkSize);
				cudaMemcpy(Md, M, chunkSize * dim * sizeof(float),
						cudaMemcpyHostToDevice);
			}

			ppt.buildKBestDB(Md, N);
//	ppt.buildKBestDB(Md, N/10);

			cout << "done buildBestDB" << endl;
			ppt.lineDist(Md, N);

			cout << "done lineDist " << endl;

			// download current result and accumulate

			cudaMemcpy(chBinPrefix, ppt.getBinPrefix(),
			HASH_SIZE * sizeof(uint), cudaMemcpyDeviceToHost);

			cudaMemcpy(chBinCounts, ppt.getBinCounts(),
			HASH_SIZE * sizeof(uint), cudaMemcpyDeviceToHost);

			cudaMemcpy(chDBIdx, ppt.getDBIdx(), chunkSize * sizeof(uint),
					cudaMemcpyDeviceToHost);

			// make sure each chunk gets the right index
			for (int i = 0; i < chunkSize; i++) {
				chDBIdx[i] += cIter * chunkSize;
			}

			cout << "merging " << endl;
			// merge the bins
			memcpy(dbIdxSave, dbIdx, base_num * sizeof(uint));

			uint accu = 0;
			for (int i = 0; i < HASH_SIZE; i++) {
				accu += chBinCounts[i];
			}
			if (accu > chunkSize) {
				cout << "Problem!!!!! accu " << accu << endl;

			}

			uint vOut = 0;
			uint vIn = 0;
			uint vChIn = 0;

			for (int i = 0; i < HASH_SIZE; i++) {

				for (int v = 0; v < binCounts[i]; v++, vIn++, vOut++)
					dbIdx[vOut] = dbIdxSave[vIn];

				for (int v = 0; v < chBinCounts[i]; v++, vChIn++, vOut++)
					dbIdx[vOut] = chDBIdx[vChIn];
			}

			// merge dbInfo
			accu = 0;
			for (int i = 0; i < HASH_SIZE; i++) {
				binCounts[i] += chBinCounts[i];
				binPrefix[i] = accu;
				accu += binCounts[i];
			}

			cout << "total number of vectors:  " << accu << endl;
//		cout << "binCounts: ";
//		for (int i = 0; i < 100; i++)
//			cout << "\t" << binCounts[i];
//		cout << endl;
//
//		cout << "dbIdx: ";
//		for (int i = 0; i < 100; i++)
//			cout << "\t" << dbIdx[i];
//		cout << endl;

			cudaMemcpy(chLines, ppt.getLine(),
					chunkSize * lineParts * sizeof(float),
					cudaMemcpyDeviceToHost);

			fLines.write((char*) chLines,
					chunkSize * lineParts * sizeof(float));
		}

		fLines.close();
		cout << "written " << lineName << endl;

		ofstream fprefix(prefixName.c_str(),
				std::ofstream::out | std::ofstream::binary);
		fprefix.write((char*) binPrefix, HASH_SIZE * sizeof(uint));
		fprefix.close();
		cout << "written " << prefixName << endl;

		ofstream fcounts(countsName.c_str(),
				std::ofstream::out | std::ofstream::binary);
		fcounts.write((char*) binCounts, HASH_SIZE * sizeof(uint));
		fcounts.close();
		cout << "written " << countsName << endl;
		cout << "size: " << (HASH_SIZE * sizeof(uint)) << endl;

		ofstream fdb(dbIdxName.c_str(),
				std::ofstream::out | std::ofstream::binary);
		fdb.write((char*) dbIdx, base_num * sizeof(uint));
		fdb.close();
		cout << "written " << dbIdxName << endl;

		if (Md)
			cudaFree(Md);

	}
		break;

	case 2: {

		if (Md)
			cudaFree(Md);
		// read data base
		ifstream fprefix(prefixName.c_str(),
				std::ifstream::in | std::ofstream::binary);
		fprefix.read((char*) binPrefix, HASH_SIZE * sizeof(uint));
		fprefix.close();
		cout << "read " << prefixName << endl;

		ifstream fcounts(countsName.c_str(),
				std::ofstream::in | std::ofstream::binary);
		fcounts.read((char*) binCounts, HASH_SIZE * sizeof(uint));
		fcounts.close();
		cout << "read " << countsName << endl;

		ifstream fdb(dbIdxName.c_str(),
				std::ifstream::in | std::ofstream::binary);
		fdb.read((char*) dbIdx, base_num * sizeof(uint));
		fdb.close();
		cout << "read " << dbIdxName << endl;

//		for (int i = 0; i < 1000; i++) {
//			cout << "\t" << dbIdx[i];
//		}
//		cout << endl;

		cout << "base_num : " << base_num << endl;
		ppt.setDB(base_num, binPrefix, binCounts, dbIdx);

		cout << "done set DB" << endl;

		vector<uint> resIdx;
		vector<float> resDist;

//		ppt.prepareDistSequence(16 * 8, 4);

//		ppt.prepareDistSequence(16 * 2, 4);

		ppt.prepare2DDistSequence(512);

		QN = gt_num;

		vector<uint> gtBins;
		gtBins.resize(QN);

		string gtBinName = baseName + "_" + intToString(dim) + "_"
				+ intToString(p) + "_" + intToString(nCluster1) + "_"
				+ intToString(nCluster2) + "_"
				+ intToString(ppt.getNPerturbations()) + ".gtBins";

		cout << "trying to read" << gtBinName << endl;
		if (!file_exists(gtBinName)) {
			vector<uint> gtVec;
			gtVec.resize(QN);

			for (int i = 0; i < QN; i++) {
				gtVec[i] = GT[i * gt_dim];
			}

			locateAll(base_num, binPrefix, binCounts, dbIdx, QN, gtVec, gtBins);

			ofstream binFile(gtBinName.c_str(),
					std::ofstream::out | std::ofstream::binary);
			binFile.write((char*) &(gtBins[0]), QN * sizeof(uint));
			binFile.close();
		} else {
			ifstream binFile(gtBinName.c_str(),
					std::ifstream::out | std::ifstream::binary);
			binFile.read((char*) &(gtBins[0]), QN * sizeof(uint));
			binFile.close();

			cout << "gtBins: " << endl;
			for (int i = 0; i < 100; i++)
				cout << "\t" << gtBins[i];
			cout << endl;
		}

		cout << "done localization" << endl;

		QN = 1000;

		for (int s = 6; s < 20; s++) {
//			int s = 11;

			resIdx.clear();
			int nVec = 2 << s;

			cout << "QN: " << QN << endl;

			uint qChunk = 100;
//			qChunk = 100;
//			qChunk = QN;
			uint qIter = QN / qChunk;

//			qIter  = 100;

			for (int i = 0; i < qIter; i++) {

				cout << " >>> " << i << " <<<<< " << endl;

				vector<uint> chResIdx;
				vector<float> chResDist;

				chResIdx.clear();

				ppt.queryBIGKNN(chResIdx, chResDist, Qd + i * qChunk * dim,
						qChunk, nVec, gtBins, i * qChunk);

//				for (int ll = 0; ll < chResIdx.size(); ll++)
//					cout << chResIdx[ll] << " ";
//				cout << endl;

				resIdx.insert(resIdx.end(), chResIdx.begin(), chResIdx.end());
			}
			analyze(resIdx, nVec, GT, gt_dim, qIter * qChunk, base_num,
					binPrefix, binCounts, dbIdx);
//			analyzeM(resIdx, nVec, gt_dim, qIter * qChunk, qOffset);
		}

	}

		break;

	case 3: {

		if (Md)
			cudaFree(Md);
		// read data base
		ifstream fprefix(prefixName.c_str(),
				std::ifstream::in | std::ofstream::binary);
		fprefix.read((char*) binPrefix, HASH_SIZE * sizeof(uint));
		fprefix.close();
		cout << "read " << prefixName << endl;

		ifstream fcounts(countsName.c_str(),
				std::ofstream::in | std::ofstream::binary);
		fcounts.read((char*) binCounts, HASH_SIZE * sizeof(uint));
		fcounts.close();
		cout << "read " << countsName << endl;

		ifstream fdb(dbIdxName.c_str(),
				std::ifstream::in | std::ofstream::binary);
		fdb.read((char*) dbIdx, base_num * sizeof(uint));
		fdb.close();
		cout << "read " << dbIdxName << endl;

//		for (int i = 0; i < 1000; i++) {
//			cout << "\t" << dbIdx[i];
//		}
//		cout << endl;

		cout << "base_num : " << base_num << endl;
		//	ppt.setDB(base_num, binPrefix, binCounts, dbIdx);

		vector<int> histogram;
		vector<float> perc;
		vector<uint> accu;
		uint bMax = 25;
		histogram.resize(bMax);
		perc.resize(bMax);
		accu.resize(bMax);

		uint low = 0;
		uint up = 1;
		uint total = 0;
		for (int b = 0; b < bMax; b++) {
			histogram[b] = 0;
			accu[b] = 0;
			up = 1 << b;
			for (int i = 0; i < HASH_SIZE; i++) {
				if ((binCounts[i] >= low) && (binCounts[i] < up)) {
					histogram[b]++;
					accu[b] += binCounts[i];
				}
			}

			total += accu[b];

			perc[b] = float(total) / float(base_num);
			if (b < 15)
				cout << low << ":  \t\t " << histogram[b] << " \t" << perc[b]
						<< " \t" << accu[b] << endl;
			else
				cout << low << ":  \t " << histogram[b] << " \t" << perc[b]
						<< " \t" << accu[b] << endl;
			low = up;
		}

	}
		break;

	case 4: {

		// lookup including line stuff

		if (Md)
			cudaFree(Md);
		// read data base
		ifstream fprefix(prefixName.c_str(),
				std::ifstream::in | std::ofstream::binary);
		fprefix.read((char*) binPrefix, HASH_SIZE * sizeof(uint));
		fprefix.close();
		cout << "read " << prefixName << endl;

		ifstream fcounts(countsName.c_str(),
				std::ofstream::in | std::ofstream::binary);
		fcounts.read((char*) binCounts, HASH_SIZE * sizeof(uint));
		fcounts.close();
		cout << "read " << countsName << endl;

		size_t nfloats = base_num;
		nfloats *= lineParts;

		float* hLines = NULL;

//		hLines = new float[nfloats];
//		cudaMallocHost(&hLines, nfloats * sizeof(float));
		// Allocate host memory using CUDA allocation calls
		cudaHostAlloc((void **) &hLines, nfloats * sizeof(float),
				cudaHostAllocMapped);

		float* dLines;
		cudaHostGetDevicePointer((void **) &dLines, (void *) hLines, 0);

		if (!hLines) {
			cerr << " did not get hLine memory " << endl;
			exit(1);
		}

		ifstream fdb(dbIdxName.c_str(),
				std::ifstream::in | std::ofstream::binary);
		fdb.read((char*) dbIdx, base_num * sizeof(uint));
		fdb.close();
		cout << "read " << dbIdxName << endl;

		// -------- sanity checks for the read data -------
		uint* hist = new uint[base_num];
		for (int i = 0; i < base_num; i++)
			hist[i] = 0;
		uint dbmin = dbIdx[0];
		uint dbmax = dbIdx[0];

		for (int i = 0; i < base_num; i++) {
			if (dbIdx[i] < dbmin)
				dbmin = dbIdx[i];
			if (dbIdx[i] > dbmax)
				dbmax = dbIdx[i];
			hist[dbIdx[i]]++;
		}

		cout << "DB min: " << dbmin << " max: " << dbmax << endl;

		dbmin = hist[0];
		dbmax = hist[0];
		uint count1 = 0;

		for (int i = 0; i < base_num; i++) {
			if (hist[i] == 1)
				count1++;

			if (hist[i] < dbmin)
				dbmin = hist[i];
			if (hist[i] > dbmax)
				dbmax = hist[i];

		}

		cout << "histmin: " << dbmin << " max: " << dbmax << endl;
		cout << "hist #1: " << count1 << endl;

		delete[] hist;

		//------------- sanity checks end -----------

		cout << "lineParts: " << lineParts << endl;
		cout << "base_num: " << base_num << endl;

#if 1
		ifstream fLines(lineName.c_str(),
				std::ifstream::in | std::ofstream::binary);

		float* hPtr = hLines;
		for (int l = 0; l < lineParts; l++, hPtr += base_num)
			fLines.read((char*) hPtr, base_num * sizeof(float));

		if (!fLines.good()) {
			cout << "something fishy!" << endl;
		}

		fLines.close();
#endif

//		for (int i = 0; i < base_num; i++) {
//
//		}
		cout << "read " << lineName << " read " << nfloats << " floats "
				<< endl;

//		for (int i = 0; i < 1000; i++) {
//			cout << "\t" << dbIdx[i];
//		}
//		cout << endl;

		cout << "base_num : " << base_num << endl;
		ppt.setDB(base_num, binPrefix, binCounts, dbIdx);

		vector<uint> resIdx;
		vector<float> resDist;

//		ppt.prepareDistSequence(16 * 8, 4);
//		ppt.prepareDistSequence(16 * 2, 4);

		ppt.prepare2DDistSequence(512);

//		for (int s = 6; s < 20; s++)
		{
			int s = 11;

			resIdx.clear();
			int nVec = 2 << s;

			cout << "QN: " << QN << endl;

			uint qChunk = 1000;
//			qChunk = 100;
//			qChunk = QN;
			uint qIter = QN / qChunk;

//			qIter  = 1;

			ppt.prepareEmptyLambda(qChunk * nVec, lineParts);

			for (int i = 0; i < qIter; i++) {

				cout << " >>> " << i << " <<<<< " << endl;

				vector<uint> chResIdx;
				vector<float> chResDist;

				chResIdx.clear();

//				ppt.queryBIGKNN(chResIdx, chResDist, Qd + i * qChunk * dim,
//						qChunk, nVec, gtBins, i * qChunk);

				ppt.queryBIGKNNRerank2(chResIdx, chResDist,
						Qd + i * qChunk * dim, qChunk, nVec, dLines);

//				for (int ll = 0; ll < chResIdx.size(); ll++)
//					cout << chResIdx[ll] << " ";
//				cout << endl;

				////////////////////////////////////////////////////////////
				// test if distances are correctly sorted?
				for (int k = 0; k < qChunk; k++) {
					bool sorted = true;
					for (int l = 1; l < nVec; l++) {
						if (chResDist[k * nVec + l - 1]
								> chResDist[k * nVec + l])
							sorted = false;
					}
					if (!sorted)
						cout << "unsorted distances at " << k << endl;
				}
				// end test sorted
				///////////////////////////////////////////////////////////

				resIdx.insert(resIdx.end(), chResIdx.begin(), chResIdx.end());
			}
			analyze(resIdx, nVec, GT, gt_dim, qIter * qChunk, base_num);
//			analyzeM(resIdx, nVec, gt_dim, qIter * qChunk, qOffset);

		}

		cudaFreeHost(hLines);

	}

		break;

	case 5:
	case 6: {

		// building the data base
		uint* dbIdxSave = new uint[base_num];

		memset(binPrefix, 0, HASH_SIZE * sizeof(uint));
		memset(binCounts, 0, HASH_SIZE * sizeof(uint));
		memset(dbIdx, 0, base_num * sizeof(uint));

		uint* chBinPrefix = new uint[HASH_SIZE];
		uint* chBinCounts = new uint[HASH_SIZE];
		uint* chDBIdx = new uint[chunkSize];
		float* chLines = new float[chunkSize * lineParts];

		ofstream fLines(lineName.c_str(),
				std::ofstream::out | std::ofstream::binary);

//		chunkNIter = 1;

		for (int cIter = 0; cIter < chunkNIter; cIter++) {

			cout << ">>>  " << cIter << " <<<<<<<<<<<< " << endl;

			if (cIter > 0) {
				delete[] M;

				M = readFloat(path_base, base_dim, chunkSize,
						cIter * chunkSize);
				cudaMemcpy(Md, M, chunkSize * dim * sizeof(float),
						cudaMemcpyHostToDevice);
			}

			ppt.buildKBestDBSparse(Md, N, sparse);

			cout << "done buildBestDBSparse" << endl;
			ppt.lineDist(Md, N);

			cout << "done lineDist " << endl;

			// download current result and accumulate

			cudaMemcpy(chBinPrefix, ppt.getBinPrefix(),
			HASH_SIZE * sizeof(uint), cudaMemcpyDeviceToHost);

			cudaMemcpy(chBinCounts, ppt.getBinCounts(),
			HASH_SIZE * sizeof(uint), cudaMemcpyDeviceToHost);

			cudaMemcpy(chDBIdx, ppt.getDBIdx(), chunkSize * sizeof(uint),
					cudaMemcpyDeviceToHost);

			// make sure each chunk gets the right index
			for (int i = 0; i < chunkSize; i++) {
				chDBIdx[i] += cIter * chunkSize;
			}

			cout << "merging " << endl;
			// merge the bins
			memcpy(dbIdxSave, dbIdx, base_num * sizeof(uint));

			uint accu = 0;
			for (int i = 0; i < HASH_SIZE; i++) {
				accu += chBinCounts[i];
			}
			if (accu > chunkSize) {
				cout << "Problem!!!!! accu " << accu << endl;

			}

			uint vOut = 0;
			uint vIn = 0;
			uint vChIn = 0;

			for (int i = 0; i < HASH_SIZE; i++) {

				for (int v = 0; v < binCounts[i]; v++, vIn++, vOut++)
					dbIdx[vOut] = dbIdxSave[vIn];

				for (int v = 0; v < chBinCounts[i]; v++, vChIn++, vOut++)
					dbIdx[vOut] = chDBIdx[vChIn];
			}

			// merge dbInfo
			accu = 0;
			for (int i = 0; i < HASH_SIZE; i++) {
				binCounts[i] += chBinCounts[i];
				binPrefix[i] = accu;
				accu += binCounts[i];
			}

			cout << "total number of vectors:  " << accu << endl;
//		cout << "binCounts: ";
//		for (int i = 0; i < 100; i++)
//			cout << "\t" << binCounts[i];
//		cout << endl;
//
//		cout << "dbIdx: ";
//		for (int i = 0; i < 100; i++)
//			cout << "\t" << dbIdx[i];
//		cout << endl;

			cudaMemcpy(chLines, ppt.getLine(),
					chunkSize * lineParts * sizeof(float),
					cudaMemcpyDeviceToHost);

			fLines.write((char*) chLines,
					chunkSize * lineParts * sizeof(float));
		}

		fLines.close();
		cout << "written " << lineName << endl;

		ofstream fprefix(prefixName.c_str(),
				std::ofstream::out | std::ofstream::binary);
		fprefix.write((char*) binPrefix, HASH_SIZE * sizeof(uint));
		fprefix.close();
		cout << "written " << prefixName << endl;

		ofstream fcounts(countsName.c_str(),
				std::ofstream::out | std::ofstream::binary);
		fcounts.write((char*) binCounts, HASH_SIZE * sizeof(uint));
		fcounts.close();
		cout << "written " << countsName << endl;
		cout << "size: " << (HASH_SIZE * sizeof(uint)) << endl;

		ofstream fdb(dbIdxName.c_str(),
				std::ofstream::out | std::ofstream::binary);
		fdb.write((char*) dbIdx, base_num * sizeof(uint));
		fdb.close();
		cout << "written " << dbIdxName << endl;

		if (Md)
			cudaFree(Md);

	}
		break;

	case 7: {

		// lookup with perfect reranking

		if (Md)
			cudaFree(Md);

		// read data base
		ifstream fprefix(prefixName.c_str(),
				std::ifstream::in | std::ofstream::binary);
		fprefix.read((char*) binPrefix, HASH_SIZE * sizeof(uint));
		fprefix.close();
		cout << "read " << prefixName << endl;

		ifstream fcounts(countsName.c_str(),
				std::ofstream::in | std::ofstream::binary);
		fcounts.read((char*) binCounts, HASH_SIZE * sizeof(uint));
		fcounts.close();
		cout << "read " << countsName << endl;

		ifstream fdb(dbIdxName.c_str(),
				std::ifstream::in | std::ofstream::binary);
		fdb.read((char*) dbIdx, base_num * sizeof(uint));
		fdb.close();
		cout << "read " << dbIdxName << endl;

		size_t nfloats = base_num;
		nfloats *= 128;

		uint8_t* hLines = NULL;

		// Allocate host memory using CUDA allocation calls
		cudaHostAlloc((void **) &hLines, nfloats * sizeof(uint8_t),
				cudaHostAllocMapped);

		float* dLines;
		cudaHostGetDevicePointer((void **) &dLines, (void *) hLines, 0);

		if (!hLines) {
			cerr << " did not get hLine memory " << endl;
			exit(1);
		}

		cout << "base_num: " << base_num << endl;

		for (int cIter = 0; cIter < chunkNIter; cIter++) {
			cout << ">>>  " << cIter << " <<<<<<<<<<<< " << endl;
			read<uint8_t>(path_base, hLines + cIter * chunkSize * dim, chunkSize * dim,
					cIter * chunkSize * dim);
		}

		for (int cIter = 0; cIter < chunkNIter; cIter++) {
			cout << cIter << ": ";
			for (int i = 0; i < 10; i++) {
//				cout << "\t" << (uint)*(hLines + cIter * chunkSize + i);
				cout << "\t"
						<< (uint) *(hLines + cIter * chunkSize * dim + chunkSize * dim- 10
								+ i);

			}
			cout << endl;
		}

		cout << "read all vectors" << endl;

		cout << "base_num : " << base_num << endl;
		ppt.setDB(base_num, binPrefix, binCounts, dbIdx);

		vector<uint> resIdx;
		vector<float> resDist;

//		ppt.prepareDistSequence(16 * 8, 4);
//		ppt.prepareDistSequence(16 * 2, 4);

		ppt.prepare2DDistSequence(512);

		cout << "example vector " << endl;
		for (int i = 0; i < 10; i++) {
			//				cout << "\t" << (uint)*(hLines + cIter * chunkSize + i);
			cout << "\t" << (uint) *(hLines + 344173742 * dim + i);

		}
		cout << endl;

//		for (int s = 6; s < 20; s++)
		{
			int s = 11;

			resIdx.clear();
			int nVec = 2 << s;

			cout << "QN: " << QN << endl;

			uint qChunk = 1000;
//			qChunk = 100;
//			qChunk = QN;
			uint qIter = QN / qChunk;

//			qIter  = 1;

			ppt.prepareEmptyLambda(qChunk * nVec);

			for (int i = 0; i < qIter; i++) {

				cout << " >>> " << i << " <<<<< " << endl;

				vector<uint> chResIdx;
				vector<float> chResDist;

				chResIdx.clear();

				ppt.queryBIGKNNRerankPerfect(chResIdx, chResDist,
						Qd + i * qChunk * dim, qChunk, nVec, dLines);

//				for (int ll = 0; ll < chResIdx.size(); ll++)
//					cout << chResIdx[ll] << " ";
//				cout << endl;

				resIdx.insert(resIdx.end(), chResIdx.begin(), chResIdx.end());
			}

			cout << "done" << endl;

			analyze(resIdx, nVec, GT, gt_dim, qIter * qChunk, base_num);

		}

		cudaFreeHost(hLines);
	}

		break;

	};

//
//	ppt.buildKBestLineDB(Md, N);

	return 0;

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
