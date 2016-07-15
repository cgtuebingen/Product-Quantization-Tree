#ifndef NEARESTNEIGHBOR_TESTVQ_C
#define NEARESTNEIGHBOR_TESTVQ_C
#include <stdio.h>
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
#include "pgrbar.hpp"

#include <sys/stat.h>
#include <sstream>
#include <iomanip>

using namespace std;


using namespace nearestNeighbor;
using namespace nearestNeighborPQ;

typedef float Dtype;
typedef unsigned char uint8_t;

bool file_exists(const std::string& _name) {
	struct stat buffer;

	return (stat(_name.c_str(), &buffer) == 0);
}

// template<typename T>
// void importData(const std::string fn, T* data, const int N) {
//     std::ifstream hnd(fn.c_str(), std::ios_base::in | std::ios_base::binary);
//     if (!hnd.good()) {
//         hnd.close();
//         throw std::runtime_error("Dataset file " + fn + " does not exists");
//     }
//     hnd.seekg( 0, std::ios::beg );
//     hnd.read(reinterpret_cast<char *>(data), sizeof(T)*N);
//     hnd.close();
// }

string intToString(const uint _x) {

	stringstream sstr;

	sstr << _x;
	return sstr.str();
}

typedef struct {
	std::string filename; // path to file
	unsigned int num;     // number of frames
	unsigned int size;    // number of sift features

	Dtype *prefix;        // array to prefix sum
	Dtype *sift;          // array to sift features
} video;

// row-major
void loadSift(std::fstream &fhnd, Dtype *sift, unsigned int len) {
	unsigned char temp_u8[sizeof(uint8_t)];
	pgrbar progressBar(len);
	for (int n = 0; n < len; ++n) {
		for (int d = 0; d < 128; ++d) {
			fhnd.read(reinterpret_cast<char *>(temp_u8), sizeof(uint8_t));
			sift[n * 128 + d] = static_cast<Dtype>(*temp_u8);
		}
		progressBar.step(n);
	}

}

int extract(video &A, video &B) {

	unsigned int temp_u32;

	const int dimSift = 128;

	std::fstream fprefixA  ((A.filename + ".prefix").c_str(), std::ios_base::in | std::ios_base::binary);
	std::fstream fprefixB  ((B.filename + ".prefix").c_str(), std::ios_base::in | std::ios_base::binary);
	std::fstream fsiftA    ((A.filename + ".sift").c_str(), std::ios_base::in | std::ios_base::binary);
	std::fstream fsiftB    ((B.filename + ".sift").c_str(), std::ios_base::in | std::ios_base::binary);
	std::fstream fmatching ((A.filename + "_" + B.filename + ".matching").c_str(), std::ios_base::in);

	// compute number of sift features
	A.prefix = new Dtype[A.num];
	printf("load information for file %s\n", A.filename.c_str());
	printf("  - total frames    %i\n", A.num);
	for (int i = 0; i < A.num; ++i) {
		fprefixA.read(reinterpret_cast<char *>(&temp_u32), sizeof(unsigned int));
		A.size += temp_u32;
		A.prefix[i] = temp_u32;
	}
	printf("  - total features: %i\n", A.size);
	A.sift   = new Dtype[A.size * dimSift];
	loadSift(fsiftA, A.sift, A.size);

	// compute number of sift features
	B.prefix = new Dtype[B.num];
	printf("load information for file %s\n", B.filename.c_str());
	printf("  - total frames    %i\n", B.num);
	for (int i = 0; i < B.num; ++i) {
		fprefixB.read(reinterpret_cast<char *>(&temp_u32), sizeof(unsigned int));
		B.size += temp_u32;
		B.prefix[i] = temp_u32;
	}
	printf("  - total features: %i\n", B.size);
	B.sift   = new Dtype[B.size * dimSift];
	loadSift(fsiftB, B.sift, B.size);



}

int main(int argc, char* argv[]) {

	// ./Ubuntu14.04/testSync /graphics/projects/scratch/wieschol/sync/2012-04-20/3D_L0001_noaudio.mp40_20_10000 100 /graphics/projects/scratch/wieschol/sync/2012-04-25/3D_L0001_noaudio.mp40_20_10000 100

	if (argc != 5) {
		printf("USAGE: %s features1 num features2 num\n\n", argv[0] );
		printf("converts two feature informations into similarity matrix  (files without endings)\n\n");
		return -1;
	}


	int c = 1;

	video vidA;
	video vidB;

	vidA.filename    = argv[c++];
	vidA.num         = atoi(argv[c++]);
	vidA.size        = 0;
	vidB.filename    = argv[c++];
	vidB.num         = atoi(argv[c++]);
	vidB.size        = 0;



	extract(vidA, vidB);

	std::cout << vidA.size % 128 << std::endl;
	std::cout << vidB.size % 128 << std::endl;
	 return 0;


	// std::cout << vidA.num << std::endl;
	// std::cout << vidB.num << std::endl;
	// return 0;

	// for (int i = 0; i < 10; ++i) {
	// 	std::cout << vidA.sift[i] << "\t" << vidB.sift[i] << std::endl;

	// }

	// return 0;

	int deviceIdx = 0;
	cudaSetDevice( deviceIdx );


	int acount = 1;

	string uniformName = "NN-data/cb";
	string baseName = string(argv[acount++]);

	uint dim, numBase, numQuery, nCluster1, nCluster2;
	int p;

	dim       = 128;
	p         = 2;
	nCluster1 = 16;
	nCluster2 = 8;

	float *dataBase, *dataQuery;


	numBase = vidB.size;
	numQuery = vidA.size;

	// generate random vectors
	dataBase = vidB.sift;;
	dataQuery = vidA.sift;

	cout << "created vectors" << endl;

	float *Md, *Qd;
	float *Distd;
	int *closestd;

	cudaMalloc(&Md, numBase * dim * sizeof(float));
	cudaMalloc(&Qd, numQuery * dim * sizeof(float));
	cudaMalloc(&closestd, numQuery * sizeof(int));

	cudaMemcpy(Md, dataBase,   numBase * dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Qd, dataQuery, numQuery * dim * sizeof(float), cudaMemcpyHostToDevice);

	int k = 16;

	PerturbationProTree ppt(dim, p, p, 1);
	// ppt.testSortLarge();
	// ppt.testDistances();

	string cbName = baseName + "_" + intToString(dim) + "_" + intToString(p)
	                + "_" + intToString(nCluster1) + "_" + intToString(nCluster2) + "_"
	                + intToString(ppt.getNPerturbations()) + ".ppqt";

	cout << "trying to read" << cbName << endl;

	if (!file_exists(cbName)) {
		ppt.createTree(nCluster1, nCluster2, Md, 300000);
		ppt.writeTreeToFile(cbName);
	} else {
		ppt.readTreeFromFile(cbName);
	}


	ppt.buildKBestDB(Md, numBase);
	ppt.lineDist(Md, numBase);



	uint nVec = 4096;
	const float ratio = 0.6;

	float *hist2d = new float[vidA.num*vidB.num];
	for (int i = 0; i < vidA.num*vidB.num; ++i)
	{
		hist2d[i] = 0;
	}

	size_t offset = 0;
	for (int fIdxA = 0; fIdxA < vidA.num; ++fIdxA) {

		printf("frame %i\n", fIdxA);
		vector<uint> resIdx;
		vector<float> resDist;

		cudaMemcpy(Qd, vidA.sift + offset, 128 * vidA.prefix[fIdxA] * sizeof(float), cudaMemcpyHostToDevice);
		offset += 128 * vidA.prefix[fIdxA];

		ppt.queryKNN(resIdx, resDist, Qd, vidA.prefix[fIdxA], nVec);


		for (int r = 0; r < vidA.prefix[fIdxA]; ++r) {
			// get vector idx
			const int IDX = resIdx[nVec*r];
			// get vector idx to frame idx
			int frameIdx = 0;
			int testIdx = 0;
			while(testIdx<IDX){
				testIdx += vidB.prefix[frameIdx];
				frameIdx++;
			}

			// if (resDist[nVec*r] < ratio * resDist[nVec*r+1]) {
				hist2d[fIdxA*vidB.num +  frameIdx]++;
			// }
		}

	}


	std::fstream fin("simi_pqt.ubin", std::ios_base::out | std::ios_base::binary);
  fin.write((char*) hist2d, vidA.num*vidB.num * sizeof(float));
  fin.close();



	cudaFree(Md);

	cudaFree(closestd);
	cudaFree(Qd);

	delete[] dataQuery;
	delete[] dataBase;

	cout << endl << "done" << endl;
	cout.flush();

}

#endif /* NEARESTNEIGHBOR_TESTVQ_C */
