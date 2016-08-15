#ifndef PRODUCTQUANTIZER_HPP
#define PRODUCTQUANTIZER_HPP

#include <Eigen/Dense>
#include "../iterator/iterator.hpp"

/**
 * @brief well known product quantizer
 */
template<typename T, uint D, uint C, uint P>
class productquantizer {

    typedef Eigen::Matrix < T, D / P, 1 > vSegment_t;

    static_assert(D % P == 0, "dim = 0 mod p failed");
    static_assert(C % 2 == 0, "cells = 0 mod 2 failed");

public:


    // ================================================================================
    // Methods
    // ================================================================================

    productquantizer() {
        _raw_centroids = new T[C * D];
        _centroids = Eigen::Map<Eigen::Matrix<T, C, D>>(_raw_centroids, C, D);
    }

    ~productquantizer() {
        delete[] _raw_centroids;
    }

    /**
     * @brief expectation step of Lloyd-iteration
     * @details assign each given vector from the iterator to coarse cluster part-wise
     *
     * @param iter [description]
     */
    void getAssignment(iterator<T, D> &iter) {
        // for each vector ...
        for (uint n = 0, n_e = iter.num(); n < n_e; ++n) {
            // for each part ...
            for (uint p = 0; p < P; ++p) {
                // extract part of base_vector
                vSegment_t vec = iter[n].segment(p * S, S);

                // find minimum
                uint bestIdx = 0;
                T bestDist = HUGE_VAL;

                // for each cluster
                for (uint c = 0, c_e = _step; c < c_e; ++c) {
                    vSegment_t cec = _centroids.row(c).segment(p * S, S);
                    T curDist = (vec - cec).squaredNorm();
                    if ( curDist  < bestDist ) {
                        bestDist = curDist;
                        bestIdx = c;
                    }
                }

                _mapping[n * P + p] = bestIdx;
                _distances[n * P + p] = bestDist;
            }
        }
    }

    /**
     * @brief maximization step of Lloyd-iteration
     * @details centroids should be the mean of all cluster vectors
     *
     * @param iter [description]
     */
    void updateCentroids(iterator<T, D> &iter) {
        _centroids = Eigen::Matrix<T, C, D>::Zero(C, D);
        T centerCounter[C * P] = {0};
        // find mean
        for (uint n = 0, n_e = iter.num(); n < n_e; ++n) {
            for (uint p = 0; p < P; ++p) {
                const uint c = _mapping[n * P + p];
                _centroids.row(c).segment(p * S, S) += iter[n].segment(p * S, S);
                ++centerCounter[p * C + c ];
            }
        }
        for (uint c = 0; c < C; ++c) {
            for (uint p = 0; p < P; ++p) {
                if (centerCounter[p * C + c] != 0)
                    _centroids.row(c).segment(p * S, S).array() /= static_cast<T>(centerCounter[p * C + c]);
            }
        }
    }

    /**
     * @brief split each centroid +- eps
     * @details [long description]
     */
    void augmentCentroids() {
        for (uint i = 0; i < _step; ++i) {
            _centroids.row(i + _step) = _centroids.row(i).array() + 0.001;
            _centroids.row(i) = _centroids.row(i).array() - 0.001;
        }
        _step *= 2;
    }

    /**
     * @brief compute squared loss as distorsion distances
     * @details [long description]
     *
     * @param iter [description]
     * @return squared loss
     */
    T loss(iterator<T, D> &iter) {
        T sum = 0;
        const uint N = iter.num() * P;

        #pragma omp parallel for reduction(+:sum)
        for (uint i = 0; i < N; ++i) {
            sum += _distances[i];
        }
        return sum;
        // Eigen::MatrixXf l = Eigen::Map<  Eigen::MatrixXf >(_distances, iter.num(), P);
        // return l.sum();
    }

    /**
     * @brief start k-means
     * @details [long description]
     *
     * @param iter vector collection that be be clusterized
     */
    void generate(iterator<T, D> &iter) {
        _step = 1;

        _mapping = new uint8_t[iter.num()*P]();
        _distances = new T[iter.num()*P]();

        
        _centroids.row(0) = iter.center();

        T currentLoss = 0.0;
        T lastLoss    = 0.0;


        do {
            uint run = 1000;
            augmentCentroids();
            do {
                lastLoss = currentLoss;
                getAssignment(iter);    // E step
                updateCentroids(iter);  // M step
                currentLoss = loss(iter);
                //std::cout << "loss "<<currentLoss<<std::endl;
                run--;
            } while (  (abs(lastLoss - currentLoss) >   0.005) && (run > 0) );
        } while (_step < C );


    }

    // ================================================================================
    // Variables
    // ================================================================================

    const uint S = D / P;
    T* _raw_centroids;
    Eigen::Matrix<T, C, D> _centroids;  // centroids of all clusters
    uint _step;                         // current number of centroids
    uint8_t *_mapping;                  // current mapping of all vectors from iter
    T *_distances;                      // current distorsion sq. distances between iter and centroids

};

#endif