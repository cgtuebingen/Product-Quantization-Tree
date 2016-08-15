#ifndef FILEHELPER_HPP
#define FILEHELPER_HPP

#include <string>
#include <iostream>

/* read Header from fvecs */
template<typename T>
T* readJegou(const char *path, uint &n, uint &d) {
    // rewrite of http://corpus-texmex.irisa.fr/fvecs_read.m
    // read little endian

    // read binary from file
    std::ifstream in(path, std::ios_base::in | std::ios_base::binary);
    std::string pp = path;
    if (!in.good()) {
        in.close();
        throw std::runtime_error("Dataset file " + pp + " does not exists");
    }

    // // get dimension of vector
    int dimension = 0;
    in.read(reinterpret_cast<char *>(&dimension), sizeof(int));

    uint vecsizeof = 1 * 4 + dimension * 4;

    // // get size of file
    in.seekg( 0, std::ios::beg );
    std::streampos fsize = in.tellg();
    in.seekg( 0, std::ios::end );
    fsize = in.tellg() - fsize;

    uint number_of_vectors = fsize / vecsizeof;
    in.seekg( 0, std::ios::beg );
    

    //cout << "dataset:    " << pp  << endl;
    //cout << "num:        " << number_of_vectors  << endl;
    //cout << "dim:        " << dimension  << endl;

    
    unsigned char temp[sizeof(T)];

    T *ptr2 = new T[number_of_vectors*dimension];

    n = number_of_vectors;
    d = dimension;

    uint pos = 0;
    uint j_end = number_of_vectors;
    for (uint j = 0; j < j_end; ++j) {

        // adjust offset
        in.read(reinterpret_cast<char *>(temp), sizeof(T));
        // read each coordinate
        for (int i = 0; i < dimension; ++i) {
            in.read(reinterpret_cast<char *>(temp), sizeof(T));
            ptr2[pos] = reinterpret_cast<T &>(temp);
            pos++;
        }


    }

    in.close();
    return ptr2;
}

template<typename T>
void readJegouHeader(const char *path, uint &n, uint &d) {
    // rewrite of http://corpus-texmex.irisa.fr/fvecs_read.m
    // read little endian

    // read binary from file
    std::ifstream in(path, std::ios_base::in | std::ios_base::binary);
    std::string pp = path;
    if (!in.good()) {
        in.close();
        throw std::runtime_error("Dataset file " + pp + " does not exists");
    }

    // // get dimension of vector
    int dimension = 0;
    in.read(reinterpret_cast<char *>(&dimension), sizeof(int));

    uint vecsizeof = 1 * 4 + dimension * 4;

    // // get size of file
    in.seekg( 0, std::ios::beg );
    std::streampos fsize = in.tellg();
    in.seekg( 0, std::ios::end );
    fsize = in.tellg() - fsize;

    uint number_of_vectors = fsize / vecsizeof;
    in.seekg( 0, std::ios::beg );
    

    n = number_of_vectors;
    d = dimension;

}

template<>
uint8_t* readJegou<uint8_t>(const char *path, uint &n, uint &d) {
    // rewrite of http://corpus-texmex.irisa.fr/fvecs_read.m
    // read little endian

    // read binary from file
    std::ifstream in(path, std::ios_base::in | std::ios_base::binary);
    std::string pp = path;
    if (!in.good()) {
        in.close();
        throw std::runtime_error("Dataset file " + pp + " does not exists");
    }

    // // get dimension of vector
    int dimension = 0;
    in.read(reinterpret_cast<char *>(&dimension), sizeof(int));

    uint vecsizeof = 4 + dimension ;

    // // get size of file
    in.seekg( 0, std::ios::beg );
    std::streampos fsize = in.tellg();
    in.seekg( 0, std::ios::end );
    fsize = in.tellg() - fsize;

    uint number_of_vectors = fsize / vecsizeof;
    in.seekg( 0, std::ios::beg );
    

    std::cout << "dataset:    " << pp  << std::endl;
    std::cout << "num:        " << number_of_vectors  << std::endl;
    std::cout << "dim:        " << dimension  << std::endl;

    n = number_of_vectors;
    d = dimension;


    
    unsigned char temp[sizeof(uint8_t)];

    uint8_t *ptr2 = new uint8_t[number_of_vectors*dimension];

    
    uint pos = 0;
    uint j_end = number_of_vectors;
    for (uint j = 0; j < j_end; ++j) {
      in.seekg((j) * sizeof(uint8_t)*132 +  sizeof(uint8_t)*4);
        // adjust offset
        //in.read(reinterpret_cast<char *>(temp), sizeof(uint8_t));
        // read each coordinate
        for (int i = 0; i < dimension; ++i) {
            in.read(reinterpret_cast<char *>(temp), sizeof(uint8_t));
            ptr2[pos] = reinterpret_cast<uint8_t &>(temp);
            pos++;
        }


    }

    in.close();
    return ptr2;
}


uint8_t* readBatchJegou(const char *path, uint start_pos, uint num) {
    // rewrite of http://corpus-texmex.irisa.fr/fvecs_read.m
    // read little endian

    // read binary from file
    std::ifstream in(path, std::ios_base::in | std::ios_base::binary);
    std::string pp = path;
    if (!in.good()) {
        in.close();
        throw std::runtime_error("Dataset file " + pp + " does not exists");
    }

    // // get dimension of vector
    int dimension = 0;
    in.read(reinterpret_cast<char *>(&dimension), sizeof(int));

    uint vecsizeof = 4 + dimension ;

    // // get size of file
    in.seekg( 0, std::ios::beg );
    std::streampos fsize = in.tellg();
    in.seekg( 0, std::ios::end );
    fsize = in.tellg() - fsize;

    uint number_of_vectors = fsize / vecsizeof;
    in.seekg( 0, std::ios::beg );
    

    // std::cout << "dataset:    " << pp  << std::endl;
    // std::cout << "num:        " << number_of_vectors  << std::endl;
    // std::cout << "dim:        " << dimension  << std::endl;



    
    unsigned char temp[sizeof(uint8_t)];

    uint8_t *ptr2 = new uint8_t[num*dimension];
    uint pos = 0;
    for (uint j = start_pos; j < start_pos+num; ++j) {
      in.seekg((j) * sizeof(uint8_t)*132 +  sizeof(uint8_t)*4);
        // adjust offset
        //in.read(reinterpret_cast<char *>(temp), sizeof(uint8_t));
        // read each coordinate
        for (int i = 0; i < dimension; ++i) {
            in.read(reinterpret_cast<char *>(temp), sizeof(uint8_t));
            ptr2[pos] = reinterpret_cast<uint8_t &>(temp);
            pos++;
        }


    }

    in.close();
    return ptr2;
}

template<>
void readJegouHeader<uint8_t>(const char *path, uint &n, uint &d) {
    // rewrite of http://corpus-texmex.irisa.fr/fvecs_read.m
    // read little endian

    // read binary from file
    std::ifstream in(path, std::ios_base::in | std::ios_base::binary);
    std::string pp = path;
    if (!in.good()) {
        in.close();
        throw std::runtime_error("Dataset file " + pp + " does not exists");
    }

    // // get dimension of vector
    int dimension = 0;
    in.read(reinterpret_cast<char *>(&dimension), sizeof(int));

    uint vecsizeof  = 4 + dimension ;

    // // get size of file
    in.seekg( 0, std::ios::beg );
    std::streampos fsize = in.tellg();
    in.seekg( 0, std::ios::end );
    fsize = in.tellg() - fsize;

    uint number_of_vectors = fsize / vecsizeof;
    in.seekg( 0, std::ios::beg );
    

    n = number_of_vectors;
    d = dimension;

}



template<typename T = uint8_t>
void write(std::string fs, uint num, uint dim, T *ptr, uint len, uint offset = 0) {
    std::fstream fin(fs.c_str(), std::ios_base::out | std::ios_base::binary);

    if (!fin.good()) {
        fin.close();
        throw std::runtime_error("write error");
    }

    if(offset == 0){
      // start from start
      fin << num << std::endl;
      fin << dim << std::endl;
      fin.ignore();
      fin.seekg( 0, std::ios::beg );
      fin.seekg( 20 + sizeof(T)*offset, std::ios::beg );
      fin.write((char*) ptr, len * sizeof(T));
    }else{
      // start somewhere
      // fin.seekg( 0, std::ios::beg );
      // fin.seekg( 20 + sizeof(T)*offset, std::ios::beg );
      fin.write((char*) ptr, len * sizeof(T));
    }

    
    fin.close();

}




template<typename T = uint8_t>
void read(std::string fs, uint &num, uint &dim, T *ptr, uint len, uint offset = 0) {
    std::ifstream fin(fs.c_str(), std::ios_base::in | std::ios_base::binary);

    if (!fin.good()) {
        fin.close();
        throw std::runtime_error("read error");
    }



    fin >> num;
    fin >> dim;
    fin.ignore();

    //std::cout << " file off "<< fin.tellg() << std::endl;

    fin.seekg( 0, std::ios::beg );
    fin.seekg( 20 + sizeof(T)*offset, std::ios::beg );
    fin.read((char*) ptr, (len)* sizeof(T));
    fin.close();

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


#endif