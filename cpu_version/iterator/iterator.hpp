#ifndef ITERATOR_HPP
#define ITERATOR_HPP

#include <string>
#include <Eigen/Dense>


template<typename T, int D>
class iterator {
public:

    iterator()  {
    }

    uint dim() const {return D;}
    uint num() const {return dataPtr.size();}

    // insert a consecutive batch of vectors
    void insertBatch(T *addr, uint n) {
        for (uint i = 0; i < n; ++i) {
            insert(addr + i * D, i);
        }
    }
    // insert a single vector
    void insert(T *addr, uint id) {
        dataPtr.push_back(addr);
        dataIdx.push_back(id);
    }

    T* addr(uint pos) const {
        return dataPtr[pos];
    }

    const Eigen::Ref< const Eigen::Matrix< T, D, 1>> operator[](const uint i)  {
        return get(i);
    }
    const Eigen::Ref< const Eigen::Matrix< T, D, 1>> get(const uint i)  {
        return Eigen::Map<Eigen::Matrix<T, D, 1>>(dataPtr[i]);
    }

    const Eigen::Matrix< T, D, 1>  center() {
        T *c = new T[D];
        Eigen::Map<Eigen::Matrix<T, D, 1>> avgVector(c);
        avgVector = Eigen::Matrix<T, D, 1>::Zero();

        for (uint n = 0; n < num(); ++n) {
            avgVector += get(n);
        }

        avgVector /= static_cast<T>(num());
        return avgVector;

    }

    std::vector<T*> dataPtr;
    std::vector<uint> dataIdx;


};

#endif