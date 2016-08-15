#ifndef VECTORQUANTIZER_HPP
#define VECTORQUANTIZER_HPP

#include <Eigen/Dense>
#include "../iterator/iterator.hpp"
/**
 * @brief plain old vector quantizer
 */
template<typename T, uint D, uint C>
class vectorquantizer {

    typedef Eigen::Matrix < T, D, 1 > vec_t;

    static_assert( ((C % 2 == 0) || ( C == 1)), "vectorquantizer: cells = 0 mod 2 failed or C==1");

public:

    // ================================================================================
    // Methods
    // ================================================================================

    vectorquantizer(){
        _raw_centroids = new T[C * D];
        _centroids = Eigen::Map<Eigen::Matrix<T, C, D>>(_raw_centroids, C, D);

    }

    ~vectorquantizer() {
        delete[] _raw_centroids;

    }

    void getAssignment(iterator<T, D> &iter) {
        // for each vector ...
        for (uint n = 0, n_e = iter.num(); n < n_e; ++n) {
            // find minimum
            uint bestIdx = 0;
            T bestDist = HUGE_VAL;
            vec_t vec = iter[n];
            // for each cluster
            for (uint c = 0, c_e = _step; c < c_e; ++c) {
                vec_t cec = _centroids.row(c);
                T curDist = (vec - cec).squaredNorm();
                if ( curDist  < bestDist ) {
                    bestDist = curDist;
                    bestIdx = c;
                }
            }
            _mapping[n] = bestIdx;
            _distances[n] = bestDist;
        }
    }

    void updateCentroids(iterator<T, D> &iter) {
        _centroids = Eigen::Matrix<T, C, D>::Zero(C, D);
        T centerCounter[C] = {0};
        // find mean
        for (uint n = 0, n_e = iter.num(); n < n_e; ++n) {
            const uint c = _mapping[n];
            _centroids.row(c) += iter[n];
            ++centerCounter[c];
        }
        for (uint c = 0; c < C; ++c) {
            if (centerCounter[c] != 0)
                _centroids.row(c).array() /= centerCounter[c];

        }
    }

    void augmentCentroids() {
        for (uint i = 0; i < _step; ++i) {
            _centroids.row(i + _step) = _centroids.row(i).array() + 0.001;
            _centroids.row(i) = _centroids.row(i).array() - 0.001;
        }
        _step *= 2;
    }

    T loss(iterator<T, D> &iter) {
        Eigen::MatrixXf l = Eigen::Map<  Eigen::MatrixXf >(_distances, iter.num(), 1);
        return l.sum();
    }

    uint id(const vec_t &vec) {
        // compute all l1 distances (row=part)
        for (uint c = 0; c < C; ++c) {
            const Eigen::Matrix< T, D, 1> cec = _centroids.row(c);
            const Eigen::Matrix< T, D, 1> diff = vec - cec;
            _L1distances[c] = diff.squaredNorm();
            _L1order[c] = c;

        }

        // now sort them according their distances (sort each row)
        auto comparator = [&](const uint8_t &lhs, const uint8_t &rhs) -> bool {
            return _L1distances[lhs] < _L1distances[rhs];
        };

        std::sort(_L1order, _L1order + C, comparator);

        return _L1order[0];

    }

    void dist(const vec_t &vec) {
        // compute all l1 distances (row=part)
        for (uint c = 0; c < C; ++c) {
            const Eigen::Matrix< T, D, 1> cec = _centroids.row(c);
            const Eigen::Matrix< T, D, 1> diff = vec - cec;
            _L1distances[c] = diff.squaredNorm();
            _L1order[c] = c;

        }


    }

    void generate(iterator<T, D> &iter) {
        _step = 1;

        _mapping = new uint8_t[iter.num()]();
        _distances = new T[iter.num()]();

        _centroids.row(0) = iter.center();

        T currentLoss = 0.0;
        T lastLoss    = 0.0;

        if (C == 1) {
            getAssignment(iter);    // E step
            updateCentroids(iter);  // M step
            return;                  // do nothing
        }

        do {
            augmentCentroids();
            do {
                lastLoss = currentLoss;
                getAssignment(iter);    // E step
                updateCentroids(iter);  // M step
                currentLoss = loss(iter);
            } while ((abs(lastLoss - currentLoss) > 0.005));
        } while (_step < C );

        delete[] _mapping;
        delete[] _distances;
    }


    // ================================================================================
    // Variables
    // ================================================================================

    Eigen::Matrix<T, C, D> _centroids;
    T *_raw_centroids;
    uint _step;
    uint8_t *_mapping;
    T *_distances;

    T _L1distances[C];
    uint _L1order[C];

};

#endif