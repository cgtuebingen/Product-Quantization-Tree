#include "mex.h"
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <typeinfo>
#include <stdint.h>
#include "../quantizer/treequantizer.hpp"
#include <iostream>

const uint D = 128;   // dimension of vector
const uint P = 2;     // number of parts
const uint C1 = 16;   // number of clusters in 1st level per part
const uint C2 = 8;    // number of refinements
const uint W = 4;
const uint LP = 32;
typedef float T;
typedef treequantizer<T, D, C1, C2, P, W, LP> tree_t;
// ==========================================================================================================
class compat_typeinfo {
    const std::type_info &ti;
public:
    explicit compat_typeinfo(const std::type_info &ti): ti(ti) {}
    const char *name() const { return ti.name(); }
    const char *raw_name() const { return ti.name(); }
};
compat_typeinfo compat_typeid(const std::type_info &ti) {
    return compat_typeinfo(ti);
}
#define typeid(x) compat_typeid(typeid(x))

#define CLASS_HANDLE_SIGNATURE 0xFF00F0A5
template<class base> class class_handle
{
public:
    class_handle(base *ptr) : ptr_m(ptr), name_m(typeid(base).raw_name()) { signature_m = CLASS_HANDLE_SIGNATURE; }
    ~class_handle() { signature_m = 0; delete ptr_m; }
    bool isValid() { return ((signature_m == CLASS_HANDLE_SIGNATURE) && !strcmp(name_m.c_str(), typeid(base).raw_name())); }
    base *ptr() { return ptr_m; }

private:
    uint32_t signature_m;
    std::string name_m;
    base *ptr_m;
};

template<class base> inline mxArray *convertPtr2Mat(base *ptr)
{
    mexLock();
    mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(new class_handle<base>(ptr));
    return out;
}

template<class base> inline class_handle<base> *convertMat2HandlePtr(const mxArray *in)
{
    if (mxGetNumberOfElements(in) != 1 || mxGetClassID(in) != mxUINT64_CLASS || mxIsComplex(in))
        mexErrMsgTxt("Input must be a real uint64 scalar.");
    class_handle<base> *ptr = reinterpret_cast<class_handle<base> *>(*((uint64_t *)mxGetData(in)));
    if (!ptr->isValid())
        mexErrMsgTxt("Handle not valid.");
    return ptr;
}

template<class base> inline base *convertMat2Ptr(const mxArray *in)
{
    return convertMat2HandlePtr<base>(in)->ptr();
}

template<class base> inline void destroyObject(const mxArray *in)
{
    delete convertMat2HandlePtr<base>(in);
    mexUnlock();
}
// ==========================================================================================================



void mexFunction(int nOutArr, mxArray *pOutArr[], int nInArr, const mxArray *pInArr[])
{
    // help info
    if (nInArr == 0) {
        std::cout << "USAGE:    pqt(cmd,arg1,arg2)"
                  << "  cmd  -  is command to run  " << std::endl
                  << "          init, run, destroy " << std::endl
                  << "EXAMPLE:  " << std::endl
                  << "          hnd = pqt('init','tree.tree','bins.bins') " << std::endl
                  << "          [id, dst] = pqt('run', hnd, q1) " << std::endl
                  << "          [id, dst] = pqt('run', hnd, q2) " << std::endl
                  << "          pqt('destroy',hnd) " << std::endl
                  <<  std::endl
                  << " id  - ids of vectors " << std::endl
                  << " dst - approximated distance " << std::endl;
        std::cout << std::endl;
        return;
    }

    // read command
    std::string cmd = mxArrayToString(pInArr[0]);

    if (cmd == "init") {
        // :: hnd = pqt('init','tree.tree','bins.bins')
        // initialize tree AND bins
        if (nInArr != 3) {
            std::string msg = "correct call is: pqt('init','*.tree','*.bins')";
            mexErrMsgIdAndTxt("pqt:init", msg.c_str());
            return;
        }

        std::string tree_path = mxArrayToString(pInArr[1]);
        std::string bin_path = mxArrayToString(pInArr[2]);

        if( !(access( tree_path.c_str(), F_OK ) != -1) ) {
            std::string errmsg = "file "+tree_path+" does not exists";
            mexErrMsgIdAndTxt("pqt:file i/o", errmsg.c_str());
        } 
        if( !(access( bin_path.c_str(), F_OK ) != -1) ) {
            std::string errmsg = "file "+bin_path+" does not exists";
            mexErrMsgIdAndTxt("pqt:file i/o", errmsg.c_str());
        } 

        tree_t *Q = new tree_t();
        Q->loadTree(tree_path);
        Q->loadBins(bin_path);
        pOutArr[0] = convertPtr2Mat<tree_t>(Q);
    } else if (cmd == "destroy") {
        // :: pqt('destroy',hnd)
        // mex-unlock
        destroyObject<tree_t>(pInArr[1]);
    } else if (cmd == "query") {
        // :: pqt('query',hnd,q)
        // get stored tree from MATLAB
        tree_t *Q = convertMat2Ptr<tree_t>(pInArr[1]);
        // read in vector
        Eigen::Map<Eigen::Matrix< T, D, 1>> q = Eigen::Map<Eigen::Matrix< T, D, 1>>((T*)mxGetData(pInArr[2]));
        // query
        std::vector<std::pair<uint, T>> vectorCandidates;    
        Q->query(20000, 500, q,  vectorCandidates);

        // return candidates
        const int i_e = vectorCandidates.size();

        pOutArr[0] = mxCreateNumericMatrix(1, i_e, mxINT32_CLASS, mxREAL);
        pOutArr[1] = mxCreateNumericMatrix(1, i_e, mxDOUBLE_CLASS, mxREAL);

        int* ids = (int *)mxGetData(pOutArr[0]);
        double* dst = (double *)mxGetData(pOutArr[1]);

        for (int i = 0; i < i_e; ++i)
        {
            ids[i] = vectorCandidates[i].first;
            dst[i] = vectorCandidates[i].second;
        }

    } else if (cmd == "build"){
        // :: pqt('build', data)
        tree_t *Q = new tree_t();
        T* learn = (T*)mxGetData(pInArr[1]);

        size_t in_rows = mxGetM(pInArr[1]);
        size_t in_cols = mxGetN(pInArr[1]);

        if(in_rows != D){
            std::string errmsg = "the vectors need to have the format d x n, where d is the dimension";
            mexErrMsgIdAndTxt("pqt:dataformat", errmsg.c_str());
        }

        iterator<float, D> iter_learn;
        iter_learn.insertBatch(learn, in_cols);
        Q->generate(iter_learn);
        pOutArr[0] = convertPtr2Mat<tree_t>(Q);

    } else if (cmd == "insert"){
        // :: pqt('insert',hnd,vec)
        #define A(i,j) A[(i) + (j)*numrows]
        // get stored tree from MATLAB
        tree_t *Q = convertMat2Ptr<tree_t>(pInArr[1]);

        T* batch = (T*)mxGetData(pInArr[2]);
        size_t in_rows = mxGetM(pInArr[2]);
        size_t in_cols = mxGetN(pInArr[2]);

        if(in_cols != D){
            std::string errmsg = "the vectors need to have the format 1 x d, where d is the dimension";
            mexErrMsgIdAndTxt("pqt:dataformat", errmsg.c_str());
        }

        Eigen::Map<Eigen::Matrix< T, D, 1>> curVec = Eigen::Map<Eigen::Matrix< T, D, 1>>(batch);
        Q->insert(curVec);

    }else if (cmd == "save_tree"){
        // :: pqt('save_tree', hnd, 'tree.tree')
        tree_t *Q = convertMat2Ptr<tree_t>(pInArr[1]);
        std::string path = mxArrayToString(pInArr[2]);
        Q->saveTree(path);

    }else if (cmd == "save_bins"){
        // :: pqt('save_bins',hnd, 'tree.bins')
        tree_t *Q = convertMat2Ptr<tree_t>(pInArr[1]);
        std::string path = mxArrayToString(pInArr[2]);
        Q->saveBins(path);

    } else if (cmd == "read_tree"){
        // :: hnd = pqt('read_tree',  'tree.tree')
        std::string path = mxArrayToString(pInArr[1]);
         tree_t *Q = new tree_t();
        Q->loadTree(path);
        pOutArr[0] = convertPtr2Mat<tree_t>(Q);

    }  else if (cmd == "read_bins"){
        // :: pqt('read_bins', hnd,  'tree.tree')
        tree_t *Q = convertMat2Ptr<tree_t>(pInArr[1]);
        std::string path = mxArrayToString(pInArr[2]);
        Q->loadBins(path);
        pOutArr[0] = convertPtr2Mat<tree_t>(Q);

    } else if (cmd == "notify"){
        // :: hnd = pqt('notify', hnd, 1000)
        int n = (int) mxGetScalar(pInArr[2]);
        tree_t *Q = convertMat2Ptr<tree_t>(pInArr[1]);
        Q->notify(n);


    }



}