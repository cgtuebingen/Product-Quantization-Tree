# PQT

This is the CPU implementation of ProductQuantizationTrees.

### Installation
You will need the [Eigen-Libray](http://eigen.tuxfamily.org/) for vector-operations. Change the `CMakeLists.txt`

    set(EIGEN_PATH "~/path/to/your/eigen")

Compiling and linking is done by

    mkdir build
    cd build
    cmake ..
    make all


### File-Layout of Vector-Files

Each `*.umem`, `*.imem`, `*.fmem` has the following layout

    uint number_of_vectors
    uint dimen_of_vector
    ... header are 20 bytes, next data start at byte 20:
    T consecutive array of data, each entry is a T

Examples

    *.umem: uint  uint 0 0 ... 0 0 uint8_t uint8_t uint8_t uint8_t uint8_t uint8_t ...  uint8_t
    *.imem: uint  uint 0 0 ... 0 0 int int int int int int ...  int
    *.fmem: uint  uint 0 0 ... 0 0 float float float float float float ...  float


### Usage

Finding the nearest neighbor consists of 3 steps (see `examples/simple.cpp`):
- build a tree (Lloyd-iterations)
- set number of maximal db-entrys (size of search space) by `notify`
- build database (insert vector-ids in bins precompute approx-distance values)
- query a vector
 
The datastructure is a template:

    treequantizer<T, D, C1, C2, P, W, LP> Quantizer;

with parameters:
- `T`  type of each coordinate
- `D`  dimension of each vector
- `C1` number of clusters in first level
- `C2` number of clusters in second level
- `P`  number of parts for indexing (number of parts)
- `W`  number of clusters to visit in first level
- `LP` number of parts for re-ranking

The query methods `Quantizer.query(...)` returns (by reference) a list of sorted vectors using the approximated distance. More precise:

    std::vector<std::pair<uint, float>> vectorCandidates; 
    // vectorCandidates[k].first is id of k-th best vector
    // vectorCandidates[k].second is approximated distance of k-th best vector

You may resort the k-th best vectors from this list exactly.

There exists a MATLAB-wrapper.