#ifndef TREEQUANTIZER_HPP
#define TREEQUANTIZER_HPP
#include <iostream>
#include <map>
#include "../helper.hpp"
#include "vectorquantizer.hpp"
#include "productquantizer.hpp"
#include "../iterator/iterator.hpp"
#include "../iterator/memiterator.hpp"


/**
 * @brief new version of product + vector quantizer
 */
template<typename T, uint D, uint C1, uint C2, uint P, uint W, uint LP>
class treequantizer {

public:


    static_assert(C1 > 0,      "C1 (number of clusters in 1st level) should be positive");
    static_assert(C2 > 0,      "C2 (number of clusters in 2nd level) should be positive");
    static_assert(D % P == 0,  "dim = 0 mod p failed (parts should divide dimension)");
    static_assert(D % LP == 0, "dim = 0 mod lp failed (line-parts should divide dimension)");
    static_assert(C1 >= W,     "C1>=W failed (pruning factor should smaller than C1)");
    static_assert(W > 0,       "W>0 failed (pruning factor should be positive)");

    typedef  std::pair< uint, T> bin_t;

    // ================================================================================
    // Methods
    // ================================================================================

    /**
     * @brief all precomputation stuff
     * @details [long description]
     */
    explicit treequantizer() {
        _currentVectorId = 0;
        _base = W * C2;
        _maxMultiIndex = pow(_base, P);
        prepareHeuristic();

        // pre calculate powers
        constexpr uint lineWidth = C1 * C2;
        _powers = new uint[P];
        for (uint p = 0; p < P; ++p) {
            _powers[p] = pow(lineWidth, p);
        }
    }

    ~treequantizer() {
        delete[] _powers;
        delete[] _multiIndexArr;
        //delete[] _dbCodes;
    }

    /**
     * @brief set the number of vectors (size of search space)
     * @details pre allocates memory for compressed information of database vectors
     *
     * @param len number of vectors in search space
     */
    void notify(size_t len) {
        _dbCodes = new code_t[len * LP];
    }

    /**
     * @brief offline computation of the heuristic, in which the bins should be traversed
     * @details
     *    all combination of bins "i \in I x I x I x I" sorted
     *    (9,2) = [[0, 0]', [1, 0]', [0, 1]', [1, 1]', [2, 0]', [2, 1]', ... ,
     *            [1, 4]', [3, 3]', ... , [9, 7]', [9, 8]', [9, 9]'']
     */
    void prepareHeuristic() {

        _multiIndexArr = new uint[_maxMultiIndex * P] ;
        uint *_multiIndexArrTmp = new uint[_maxMultiIndex * P] ;

        T *norm = new T[_maxMultiIndex];
        uint *order = new uint[_maxMultiIndex];

        // generate a tuples in {0,...,_maxMultiIndex}^P
        for (uint idx = 0; idx < _maxMultiIndex; ++idx) {

            uint decNum = idx;
            uint p = 0;

            // compute a valid index
            uint arr[P] = {0};
            T arrTmp[P] = {0};
            while (decNum > 0) {
                const uint rem =  decNum % _base;
                arr[p] = rem;
                arrTmp[p] = rem;
                decNum /= _base;
                ++p;
            }


            Eigen::Matrix<T, P, 1> sortCrit = Eigen::Map<Eigen::Matrix<T, P, 1>>(arrTmp);
            norm[idx] = sortCrit.squaredNorm();

            for (uint p = 0; p < P; ++p) {
                _multiIndexArrTmp[ idx * P + p] = arr[p];
            }
            order[idx] = idx;
        }

        // sort these indicies (tuples)
        auto comparator = [&](const uint & lhs, const uint & rhs) -> bool {
            return norm[lhs] < norm[rhs];
        };
        std::sort(order, order + _maxMultiIndex, comparator);

        // for speed reason: copy the heuristic in the correct order
        for (uint h = 0; h < _maxMultiIndex; ++h) {
            for (uint p = 0; p < P; ++p) {
                _multiIndexArr[ h * P + p] = _multiIndexArrTmp[ order[h] * P + p];
            }
        }

        // remove temporary objects
        delete[] _multiIndexArrTmp;
        delete[] norm;
        delete[] order;
    }

    /**
     * @brief group all vector segments that belong to a cluster of the first level
     * @details [long description]
     *
     * @param p current segement
     * @param c current cluster
     * @param haystack access to vector
     * @param subset new iterator
     *
     * @return iterator with just the corresponding vector addr.
     */
    void group(uint p, uint c, const iterator<T, D> &haystack, iterator < T, D / P > &subset) {
        for (uint n = 0, n_e = haystack.num(); n < n_e; ++n) {
            const uint assignIdx = _PQ->_mapping[n * P + p];
            if (assignIdx == c) {
                subset.insert(haystack.addr(n) + S * p, n);
            }
        }
    }

    /**
     * @brief cluster iter into bins of tree
     * @details [long description]
     *
     * @param iter [description]
     */
    void generate(iterator<T, D> &haystack) {
        // first level
        _PQ = new productquantizer<T, D, C1, P>();
        _PQ->generate(haystack);

        // refinement clusters
        _VQ.resize(P * C1);

        parallel_for (uint p = 0; p < P; ++p) {
            for (uint c = 0; c < C1; ++c) {
                iterator < T, D / P > subset;
                group(p, c, haystack, subset);
                vectorquantizer < T, D / P, C2 > *Q = new vectorquantizer < T, D / P, C2 > ();
                Q->generate(subset);
                _VQ[c * P + p] = Q;

            }
        }

        // prepare pairwise distances between first level centroids
        computeLookupTable();

    }

    /**
     * @brief precomputes a lookup table with all pairwise distances between first level centroids
     * @details [long description]
     */
    void computeLookupTable() {
        // compute all fine coarse distances for lookup table
        _coarseDistLookup = new T[LP * C1 * C1];

        for (uint i = 0; i < C1; ++i) {

            const Eigen::Matrix < T, D, 1 > AV = _PQ->_centroids.row(i);

            for (uint j = i; j < C1; ++j) {

                const Eigen::Matrix < T, D, 1 > BV = _PQ->_centroids.row(j);
                const Eigen::Matrix < T, D, 1 > diff  = AV - BV;

                for (uint p = 0; p < LP; ++p) {
                    const T curDistance = diff.segment(p * SS, SS).squaredNorm();
                    _coarseDistLookup[ (p * C1    + j )* C1 + i  ] = curDistance;
                    _coarseDistLookup[ (p * C1    + i )* C1 + j  ] = curDistance;
                }
            }
        }
    }

    /**
     * @brief insert a vector with identifier idx into a bin
     * @details [long description]
     *
     * @param vec vector that should be used
     * @param idx unique identifier of vector (will be returned by query)
     */
    void insert(const Eigen::Matrix< T, D, 1> &vec) {
        const uint binIdx = id(vec);
        _bins[binIdx].push_back(_currentVectorId);
        prepareReranking();
        ++_currentVectorId;
    }

    /**
     * @brief computes the global id given both level ids per part
     * @details just for illustration
     *
     * @param c coarse ids per part
     * @param f fine ids per part
     * @return global id in (C1*C2)^P
     */
    uint globId(const std::vector<uint> c, const std::vector<uint> f) {
        uint i = 0;
        for (uint p = 0; p < P; ++p) {
            i += (c[p] * C2 + f[p]) * _powers[p];
        }
        return i;
    }

    /**
     * @brief returns list of vectors without reranking
     * @details [long description]
     *
     * @param boundVectors max of used vectors
     * @param boundBins max of used bins
     * @param queryVec query vector
     * @param vectorIdx sorted list of vectors (id, approx. distance)
     * @param binCandidates unsorted bin candidate list (id, d)
     * @param seqOrder order of bins
     * @param sortBins sort bins after heuristic
     */
    void unorderedList(const uint boundVectors, const uint boundBins, const Eigen::Matrix< T, D, 1> &vec,
                       std::vector<std::pair<uint, T>> &vectorIdx,
                       std::vector<bin_t> &binCandidates,
                       std::vector<uint> &seqOrder, bool sortBins = true) {

        uint *segOrder = new uint[P * W * C2];
        uint *segL1Id  = new uint[P * W * C2];
        uint *segL2Id  = new uint[P * W * C2];
        T *segD1Id     = new T[P * W * C2];
        T *segD2Id     = new T[P * W * C2];

        // compute all intermediate values
        id(vec);
        // extract segment information from tree
        segmentInfo(vec, segL1Id, segL2Id, segD1Id, segD2Id, segOrder);
        // build bins ordered by our heuristic
        orderBins(boundBins, segL1Id, segL2Id, segD1Id, segD2Id, segOrder, binCandidates, seqOrder, sortBins);
        // extract vectors from these bins
        enumerateVectors(boundVectors,  binCandidates, seqOrder, vectorIdx);

        delete[] segOrder;
        delete[] segL1Id;
        delete[] segL2Id;
        delete[] segD1Id;
        delete[] segD2Id;
    }

    /**
     * @brief returns a list of vectors reranked
     * @details [long description]
     *
     * @param boundVectors max of used vectors
     * @param boundBins max of used bins
     * @param queryVec query vector
     * @param vectorIdx sorted list of vectors (id, approx. distance)
     * @param sortBins sort bins after heuristic
     */
    void orderedList(const uint boundVectors, const uint boundBins, const Eigen::Matrix< T, D, 1> &queryVec,
                     std::vector<std::pair<uint, T>> &vectorIdx,
                     std::vector<bin_t> &binCandidates,
                     std::vector<uint> &seqOrder, bool sortBins = true) {

        uint *segOrder = new uint[P * W * C2];
        uint *segL1Id  = new uint[P * W * C2];
        uint *segL2Id  = new uint[P * W * C2];
        T *segD1Id     = new T[P * W * C2];
        T *segD2Id     = new T[P * W * C2];

        // compute all intermediate values
        id(queryVec);
        // extract segment information from tree
        segmentInfo(queryVec, segL1Id, segL2Id, segD1Id, segD2Id, segOrder);
        // build bins ordered by our heuristic
        orderBins(boundBins, segL1Id, segL2Id, segD1Id, segD2Id, segOrder, binCandidates, seqOrder, sortBins);
        // extract vectors from these bins and re-rank them
        rerankVectors(boundVectors, binCandidates, seqOrder, vectorIdx);

        delete[] segOrder;
        delete[] segL1Id;
        delete[] segL2Id;
        delete[] segD1Id;
        delete[] segD2Id;
    }

    /**
     * @brief returns a list of nearest neighbors for a given query vector
     * @details [long description]
     *
     * @param boundVectors max of used vectors
     * @param boundBins max of used bins
     * @param queryVec query vector
     * @param vectorIdx sorted list of vectors (id, approx. distance)
     * @param binCandidates unsorted bin candidate list (id, d)
     * @param seqOrder order of bins
     * @param sortBins sort bins after heuristic
     */
    void query(const uint boundVectors, const uint boundBins, const Eigen::Matrix< T, D, 1> &queryVec,
               std::vector<std::pair<uint, T>> &vectorIdx, bool sortBins = true) {


        std::vector<bin_t> binCandidates;
        std::vector<uint> seqOrder;

        uint *segOrder = new uint[P * W * C2];
        uint *segL1Id  = new uint[P * W * C2];  // ids of first level
        uint *segL2Id  = new uint[P * W * C2];  // ids of second level
        T *segD1Id     = new T[P * W * C2];     // distances of first level
        T *segD2Id     = new T[P * W * C2];     // distances of second level

        // compute all intermediate values
        id(queryVec);
        // extract segment information from tree
        segmentInfo(queryVec, segL1Id, segL2Id, segD1Id, segD2Id, segOrder);
        // build bins ordered by our heuristic
        orderBins(boundBins, segL1Id, segL2Id, segD1Id, segD2Id, segOrder, binCandidates, seqOrder, sortBins);
        // extract vectors from these bins and re-rank them
        rerankVectors(boundVectors, binCandidates, seqOrder, vectorIdx);

        delete[] segOrder;
        delete[] segL1Id;
        delete[] segL2Id;
        delete[] segD1Id;
        delete[] segD2Id;
    }

    /**
     * @brief pre compute compressed information of each database vector
     * @details [long description]
     */
    void prepareReranking() {

        // compute ratio for each database vector
        float curQuantError = 0;
        for (uint p = 0; p < LP; ++p) {

            T best_lambda = 0;
            T best_quanterror = HUGE_VAL;
            uint best_id_A = 0;
            uint best_id_B = 0;

            for (uint A = 0; A < C1; ++A) {
                // side A <-> db
                const T side_b = _L1distancesVirtual[p * C1 + A];

                for (uint B = A + 1; B < C1; ++B) {
                    // side B <-> db
                    const T side_a = _L1distancesVirtual[p * C1 + B];
                    // side A <-> B
                    const T side_c = _coarseDistLookup[ (p * C1    + A ) * C1 + B  ];
                    // fraction of side "c" for projection
                    const T lambda = calcRatio(side_a, side_b, side_c);
                    // projection error by triangulation
                    const T quanterror = extractDistance(side_a, side_b, side_c, lambda);

                    if (quanterror < best_quanterror) {
                        best_quanterror = quanterror;
                        best_lambda = lambda;
                        best_id_A = A;
                        best_id_B = B;
                    }
                }
            }
            // compress current part-information
            _dbCodes[(size_t)(_currentVectorId * LP + p)] = code_t(best_id_A, best_id_B, best_lambda);

            curQuantError += best_quanterror;
            if (best_lambda < minLambdaValue) {
                minLambdaValue = best_lambda;
            }
            if (best_lambda > maxLambdaValue) {
                maxLambdaValue = best_lambda;
            }

        }

        avgQuantError += curQuantError / static_cast<T>( LP );

        if (curQuantError < minQuantError) {
            minQuantError = curQuantError;
        }
        if (curQuantError > maxQuantError) {
            maxQuantError = curQuantError;
        }


    }

    /**
     * @brief computes approximate distance
     * @details approximate distance between a vector "vec" and a dbVector with index "dbIdx" from bin "curBin"
     *
     * @param queryVec query vector
     * @param curBin binInformation for query vector about bin for dbVector
     * @param dbIdx id of db vector
     * @return approximate distance
     */
    T distance(size_t dbIdx) {
        T approxDist = 0;

        for (uint p = 0; p < LP; ++p) {
            const uint A = _dbCodes[(size_t)dbIdx * LP + p].a();
            const uint B = _dbCodes[(size_t)dbIdx * LP + p].b();
            const float lambda = _dbCodes[(size_t)dbIdx * LP + p].lambda();
            // get triangle sides
            const T side_b = _L1distancesVirtual[p * C1 + A];
            const T side_a = _L1distancesVirtual[p * C1 + B];
            const T side_c = _coarseDistLookup[ (p * C1    + A ) * C1 + B  ];

            approxDist += extractDistance(side_a, side_b,  side_c, lambda);
        }

        return approxDist;
    }

    /**
     * @brief sorts a list of vectors according the approximate distance to the query vector
     * @details [long description]
     *
     * @param maxVecs maximum of vectors that should be used = |L_s|
     * @param binDesc information about related bins
     * @param seqOrder visiting order of bins
     * @param vectorIdx (id,distance) pairs of candidates , sorted
     */
    void rerankVectors(const uint maxVecs, const std::vector<bin_t> &binDesc, const  std::vector<uint> &seqOrder,
                       std::vector<std::pair<uint, T>> &vectorIdx) {

        vectorIdx.reserve(maxVecs);
        uint usedVecs = 0;
        bool stopFlag = false;
        // iterate proposed bins
        for (uint b = 0, b_e = binDesc.size(); b < b_e; ++b) {

            const bin_t &curBin = binDesc[seqOrder[b]];
            const uint globBinIdx = curBin.first;
            // iterate all vectors in current proposed bin
            for (uint i = 0, i_e =  _bins[globBinIdx].size(); i < i_e; ++i) {
                const uint curVecIdx = _bins[globBinIdx][i];

                T d = distance(curVecIdx);
                vectorIdx.push_back({curVecIdx, d});

                ++usedVecs;
                // enought ?
                if (usedVecs > maxVecs) {
                    stopFlag = true;
                }

            }
            if (stopFlag)
                break;
        }

        // sort these vectors according their approximated distance
        auto comparator = [&](const std::pair<uint, T> &lhs, const std::pair<uint, T> &rhs) -> bool {
            return lhs.second < rhs.second;
        };
        std::sort(vectorIdx.begin(), vectorIdx.end(), comparator);
    }

    /**
     * @brief returns a bin histogram
     * @details [long description]
     *
     * @param bin vector containing histogram
     */
    void binHist(std::vector<uint> &bin) {

        bin.resize(5);
        for (std::map<uint, std::vector<uint>>::iterator it = _bins.begin(); it != _bins.end(); ++it) {
            const uint count = it->second.size();
            if (count > 10000)
                bin[4]++;
            else if (count > 1000)
                bin[3]++;
            else if (count > 100)
                bin[2]++;
            else if (count > 10)
                bin[1]++;
            else if (count > 1)
                bin[0]++;
        }

    }

    /**
    * @brief like re-ranking but without sorting resulting vector-ids
    * @details each distance is set to zero
    *
    * @param binDesc bin information
    * @param seqOrder order of bin traversing
    * @param vectorIdx all candidate vector ids
    */
    void enumerateVectors(const uint maxVecs,
                          const std::vector<bin_t> &binDesc,
                          std::vector<uint> &seqOrder,
                          std::vector<std::pair<uint, T>> &vectorIdx) {
        uint usedVecs = 0;
        bool stopFlag = false;
        for (uint b = 0; b < binDesc.size(); ++b) {
            const uint globBinIdx = binDesc[seqOrder[b]].first;
            for (uint s = 0, s_e = _bins[globBinIdx].size(); s < s_e; ++s) {
                vectorIdx.push_back({_bins[globBinIdx][s], 0});
                if (usedVecs > maxVecs) {
                    stopFlag = true;
                }
                ++usedVecs;
            }
            if (stopFlag)
                break;
        }
    }




    /**
     * @brief use bin info for ordering the bins
     * @details [long description]
     *
     * @param sortBins sorting for further processing (dynamic heuristic) or just precomputed statis heuristic
     */
    void orderBins(uint maxBins, uint *segL1Id, uint *segL2Id, T *segD1Id, T *segD2Id, uint *segOrder,
                   std::vector <bin_t> &binCandidates, std::vector<uint> &seqOrder, bool sortBins = true
                  ) {
        UNUSED(segD1Id);
        uint h_e = min(maxBins, _maxMultiIndex);
        binCandidates.clear();
        binCandidates.reserve(h_e);

        // ordering of each proposed bin
        seqOrder.clear();
        seqOrder.reserve(h_e);

        // iterate all candidates
        for (uint h = 0; h < h_e; ++h) {

            uint globIdx = 0;
            T curFineDist = 0;
            // remember stuff
            for (uint p = 0; p < P; ++p) {
                // compute a valid index
                const uint idx = _multiIndexArr[ h * P + p];
                const uint kkk = segOrder[p * W * C2 + idx ];

                curFineDist += segD2Id[p * W * C2 + kkk ];
                globIdx += (segL1Id[p * W * C2 + kkk ] * C2 + segL2Id[p * W * C2 + kkk ]) * _powers[p];
            }
            // save stuff
            binCandidates.push_back( {globIdx, curFineDist});
            seqOrder.push_back(h);
        }

        if (sortBins) {
            // order everything by second level distance
            auto comparator = [&](const uint & lhs, const uint & rhs) -> bool {
                return binCandidates[lhs].second < binCandidates[rhs].second;
            };
            std::sort(seqOrder.begin(), seqOrder.end(), comparator);
        }


    }

    /**
     * @brief extract all information for a query
     * @details [long description]
     *
     * @param segInfo [description]
     * @return [description]
     */
    void segmentInfo(const Eigen::Matrix< T, D, 1> &vec, uint *segmentL1Id, uint *segmentL2Id,
                     T *segmentD1Id, T *segmentD2Id, uint *segmentOrder) {

        // identify part-wise best clusters
        for (uint p = 0; p < P; ++p) {
            // now step into the best "W" of them for proposing bins
            uint pos = 0;
            for (uint h1 = 0; h1 < W; ++h1) {

                const uint c1 = _L1order[p * C1 + h1 ];
                const T l1 = _L1distances[p * C1 + c1];
                const Eigen::Matrix < T, D / P, 1 > vecseg = vec.segment(p * S, S);

                vectorquantizer < T, D / P, C2 > *Q = _VQ[c1 * P + p];
                Q->dist(vecseg);

                for (uint h2 = 0; h2 < C2; ++h2) {
                    const T l2 = Q->_L1distances[h2];
                    segmentL1Id[p * W * C2 + pos ]  = c1;
                    segmentL2Id[p * W * C2 + pos ]  = h2;
                    segmentD1Id[p * W * C2 + pos ]  = l1;
                    segmentD2Id[p * W * C2 + pos ]  = l2;
                    segmentOrder[p * W * C2 + pos ] = pos;
                    ++pos;
                }
            }

            auto comparator = [&](const uint & lhs, const uint & rhs) -> bool {
                return segmentD2Id[p * W * C2 + lhs ] < segmentD2Id[p * W * C2 + rhs ];
            };
            std::sort(segmentOrder + p * W * C2, segmentOrder + p * W * C2 + W * C2, comparator);

        }
    }


    /**
     * @brief computes id of bin from a vector (naive but fast version)
     * @details [long description]
     *
     * @param vec vector that should be sorted in
     * @return identifier of computed bin
     */
    uint id(const Eigen::Matrix< T, D, 1> &vec) {

        // compute coarse distances
        for (uint c = 0; c < C1; ++c) {
            // get current centroid
            const Eigen::Matrix< T, D, 1> cec = _PQ->_centroids.row(c);
            // compute difference
            const Eigen::Matrix< T, D, 1> diff = vec - cec;
            // for each segment extract squared distance

            for (uint p = 0; p < P; ++p) {
                _L1order[p * C1 + c] = c;

                T d = 0;
                for (uint pp = 0; pp < LP / P; ++pp) {
                    T dd = diff.segment((pp + p * LP / P  ) * SS, SS).squaredNorm();
                    _L1distancesVirtual[(pp + p * LP / P  ) * C1 + c] = dd;
                    d += dd;
                }
                _L1distances[p * C1 + c] = d;
            }
        }

        // now sort the centroids according their distances (sort each row = per part)
        T* distances;
        auto comparator = [&](const uint &lhs, const uint &rhs) -> bool {
            return distances[lhs] < distances[rhs];
        };
        for (uint p = 0; p < P; ++p) {
            distances = _L1distances + p * C1;
            std::sort(_L1order + p * C1, _L1order + p * C1 + C1, comparator);
        }

        // compute entire id
        uint i = 0;
        for (uint p = 0; p < P; ++p) {
            // extract vector part
            const Eigen::Matrix < T, D / P, 1 > vecseg = vec.segment(p * S, S);
            // best 1st level id
            const uint c = _L1order[p * C1 + 0];
            uint secondLevelId = _VQ[c * P + p]->id(vecseg);
            // step into vector quantizer
            const uint ii = ((c * C2 + secondLevelId) * _powers[p]) ;
            i += ii;
        }

        return (i);

    }
    // ================================================================================
    // Serialize
    // ================================================================================

    /**
     * @brief stores only the tree (all clusters)
     * @details [long description]
     *
     * @param fs file name
     */
    void saveTree(std::string fs) {

        std::fstream fHandle(fs.c_str(), std::ios_base::out | std::ios_base::binary);
        if (!fHandle.good()) {
            fHandle.close();
            throw std::runtime_error("write error: cannot open file " + fs);
        }

        serializeValue<uint>(fHandle, D );
        serializeValue<uint>(fHandle, C1 );
        serializeValue<uint>(fHandle, C2 );
        serializeValue<uint>(fHandle, P );
        serializeValue<uint>(fHandle, W );


        // save 1st level centroids
        for (uint c1 = 0; c1 < C1; ++c1) {
            for (uint d = 0; d < D; ++d) {
                T jj = _PQ->_centroids(c1, d);
                serializeValue(fHandle, jj);
            }
        }
        // save 2nd level centroids
        for (uint p = 0; p < P; ++p) {
            for (uint c = 0; c < C1; ++c) {
                vectorquantizer < T, D / P, C2 > *Q = _VQ[c * P + p];

                for (uint c2 = 0; c2 < C2; ++c2) {
                    for (uint i = 0; i < D / P; ++i) {
                        T jj = Q->_centroids(c2, i);
                        serializeValue<T>(fHandle, jj);
                    }
                }
            }
        }

        fHandle.close();

    }

    /**
     * @brief stores database related information
     * @details stores all bins and approx. distance (compressed) information
     *
     * @param fs [description]
     */
    void saveBins(std::string fs) {

        std::fstream fHandle(fs.c_str(), std::ios_base::out | std::ios_base::binary);
        if (!fHandle.good()) {
            fHandle.close();
            throw std::runtime_error("write error: cannot open file " + fs);
        }

        serializeValue<uint>(fHandle, _bins.size() );

        for (std::map<uint, std::vector<uint>>::iterator it = _bins.begin(); it != _bins.end(); ++it) {
            serializeValue<uint>(fHandle, it->first );
            serializeValue<uint>(fHandle, it->second.size() );
            for (uint j = 0; j < it->second.size(); ++j) {
                serializeValue<uint>(fHandle, it->second[j]);
            }

        }

        serializeValue<uint>(fHandle, _currentVectorId );
        serializeValue<uint>(fHandle, LP );

        // save lineinfo
        for (size_t i = 0; i < _currentVectorId * LP; ++i) {
            serializeValue<T>(fHandle, _dbCodes[(size_t)i].raw);
        }

        fHandle.close();

    }

    /**
     * @brief load tree
     * @details [long description]
     *
     * @param fs [description]
     */
    void loadTree(std::string fs) {

        std::fstream fHandle(fs.c_str(), std::ios_base::in | std::ios_base::binary);
        if (!fHandle.good()) {
            fHandle.close();
            throw std::runtime_error("read error: cannot open file" + fs);
        }

        uint r_D  = 0;
        uint r_C1 = 0;
        uint r_C2 = 0;
        uint r_P  = 0;
        uint r_W  = 0;

        unserializeValue<uint>(fHandle, r_D );
        unserializeValue<uint>(fHandle, r_C1 );
        unserializeValue<uint>(fHandle, r_C2 );
        unserializeValue<uint>(fHandle, r_P );
        unserializeValue<uint>(fHandle, r_W );

        if (r_D != D)      throw std::runtime_error("D missmatch");
        if (r_C1 != C1)    throw std::runtime_error("C1 missmatch");
        if (r_C2 != C2)    throw std::runtime_error("C2 missmatch");
        if (r_P != P)      throw std::runtime_error("P missmatch");

        // load 1st level centroids
        _PQ = new productquantizer<T, D, C1, P>();
        for (uint c1 = 0; c1 < C1; ++c1) {
            for (uint d = 0; d < D; ++d) {
                T jj = 0;
                unserializeValue<T>(fHandle, jj);
                _PQ->_centroids(c1, d) = jj;
            }
        }

        // load 2nd level centroids
        _VQ.resize(P * C1);
        for (uint p = 0; p < P; ++p) {
            for (uint c = 0; c < C1; ++c) {
                vectorquantizer < T, D / P, C2 > *Q = new vectorquantizer < T, D / P, C2 > ();
                _VQ[c * P + p] = Q;
                Q->_centroids = Eigen::Matrix < T, C2, D / P >::Zero(C2, D / P);
                Q->_step = C2;
                for (uint c2 = 0; c2 < C2; ++c2) {
                    for (uint i = 0; i < D / P; ++i) {
                        T jj = 0;
                        unserializeValue<T>(fHandle, jj);
                        Q->_centroids(c2, i) = jj;
                    }
                }
            }
        }

        computeLookupTable();

    }

    /**
     * @brief load compressed database vector information
     * @details [long description]
     *
     * @param fs [description]
     */
    void loadBins(std::string fs) {

        std::fstream fHandle(fs.c_str(), std::ios_base::in | std::ios_base::binary);
        if (!fHandle.good()) {
            fHandle.close();
            throw std::runtime_error("read error: cannot open file " + fs);
        }

        uint countBins = 0;
        unserializeValue<uint>(fHandle, countBins );

        _currentVectorId = 0;
        for (uint i = 0; i < countBins; ++i) {
            uint binId = 0;
            uint binSize = 0;

            unserializeValue<uint>(fHandle, binId );
            unserializeValue<uint>(fHandle, binSize );

            for (uint j = 0; j < binSize; ++j) {
                uint curId = 0;
                unserializeValue<uint>(fHandle, curId);
                _bins[binId].push_back(curId);
                _currentVectorId++;
            }


        }

        uint r_len, r_lp;

        unserializeValue<uint>(fHandle, r_len );
        unserializeValue<uint>(fHandle, r_lp );

        if (r_lp  != LP)                throw std::runtime_error("LP missmatch");
        if (r_len != _currentVectorId)  throw std::runtime_error("#vectors missmatch ");

        notify(_currentVectorId);

        // read lineinfo
        for (size_t i = 0; i < _currentVectorId * LP; ++i) {
            T t = 0;
            unserializeValue<T>(fHandle, t);
            _dbCodes[i].raw = t;
        }

        fHandle.close();

    }


    // ================================================================================
    // Variables
    // ================================================================================

    const uint S = D / P;                                       // segment length
    const uint SS = D / LP;                                     // segment length for re-ranking

    uint* _multiIndexArr;                                       // all multi-indicies
    size_t _maxMultiIndex;                                        // number of multi-indicies
    uint _base;                                                 // W*C2

    uint* _powers;                                              // precomputed powers
    code_t *_dbCodes;                                           // compressed database information

    productquantizer<T, D, C1, P> *_PQ;                         // first level
    std::vector < vectorquantizer < T, D / P, C2 > * > _VQ;     // second level

    T _L1distances[C1 * P];                                     // remember all first distance
    T _L1distancesVirtual[C1 * LP];                             // remember all first distance
    uint _L1order[C1 * P];                              // remember order of matching first distances
    std::map<uint, std::vector<uint>> _bins;                    // bin representation

    T *_coarseDistLookup;                                       // lookup table for fi


    size_t _currentVectorId;                                      // serial no. of next base vector

    double avgQuantError = 0;
    T minQuantError = HUGE_VAL;
    T maxQuantError = -1;
    T minLambdaValue = HUGE_VAL;
    T maxLambdaValue = -HUGE_VAL;

};


#endif