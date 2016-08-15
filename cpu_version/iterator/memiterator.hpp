#ifndef MEMITERATOR_HPP
#define MEMITERATOR_HPP

#include <string>
#include <fstream>
#include <Eigen/Dense>
#include "../helper.hpp"

/*
T = output format
TT = input format
D = dim
*/
template<typename T, typename TT = uint8_t>
class memiterator {
public:

    memiterator( )  {
    }

    ~memiterator() {
        _handle->close();
    }

    std::string _fs;
    std::ifstream *_handle;

    void open(std::string fs) {
        _fs = fs;
        _handle = new std::ifstream (fs.c_str(), std::ios_base::in | std::ios_base::binary);

        if (!_handle->good()) {
            _handle->close();
            throw std::runtime_error("read error for file " + fs);
        }

        *_handle >> _num;
        *_handle >> _dim;
        _handle->ignore();


    }

    uint dim() const {return _dim;}
    uint num() const {return _num;}

    // insert a consecutive batch of vectors

    T* addr(uint pos) const {

        TT* place = new TT[_dim];
        _handle->seekg( 0, std::ios::beg );
        _handle->seekg( 20 + sizeof(TT)*pos * _dim, std::ios::beg );
        _handle->read((char*) place, (_dim)* sizeof(TT));

        T* out = batchcast<TT, T>(place, _dim);
        delete[] place;
        return out;
    }

    T* all() const {
        TT* raw = new TT[_dim * _num];
        _handle->seekg( 0, std::ios::beg );
        _handle->seekg( 20, std::ios::beg );
        _handle->read((char*) raw, (_dim * _num)* sizeof(TT));


        T * data = batchcast<TT, T>(raw, _dim * _num);
        delete[] raw; // !!!
        return data;
    }

    T* all(uint len, uint offset = 0) const {
        TT* raw = new TT[_dim * _num];
        _handle->seekg( 0, std::ios::beg );
        _handle->seekg( 20 + (_dim * offset)* sizeof(TT), std::ios::beg );
        _handle->read((char*) raw, (_dim * len)* sizeof(TT));


        T * data = batchcast<TT, T>(raw, _dim * _num);
        delete[] raw; // !!!
        return data;
    }



    uint _dim;
    uint _num;

};

#endif