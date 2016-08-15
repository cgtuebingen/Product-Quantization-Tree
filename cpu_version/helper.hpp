#ifndef HELPER_HPP
#define HELPER_HPP

#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif


#include <tuple>
#include <vector>
#include <cmath>
#include <fstream>

#define UNUSED(arg) (void)arg;    //  trick of Qt

typedef unsigned int uint;

// this is faster than pow(..)
template <class T>
inline constexpr T pow(T const& x, uint n) {
    return n > 0 ? x * pow(x, n - 1) : 1;
}

template<typename T, typename TT> inline constexpr T min(T const &x, TT const &y) {
    return ( x < y ) ? x : y;
}
template<typename T, typename TT> inline constexpr T max(T const &x, TT const &y) {
    return ( x > y ) ? x : y;
}

template<typename T> inline const T abs(T const &x) {
    return ( x < 0 ) ? -x : x;
}

// contains information for each part
// for small number of clusters we can use 2 bytes (16,16) clusters 
// (1byte for lambda; 1byte for p1,p2)
// for large datasets on might use 3 bytes
typedef union code_t {


    typedef unsigned short int us;
    typedef unsigned char ui8;

    // 4bytes each part
    float raw;
    struct {
        ui8 p1;
        ui8 p2;
        us l;
    } val;

    inline us a() const {
        return val.p1;
    }
    inline us b() const {
        return val.p2;
    }

    inline float lambda() const {
        return (float(val.l) * (8.f / 65536.f) - 4.f);
    }

    inline void setLambda(float l) {
        val.l = toUShort(l);
    }
    inline void setA(int l) {
        val.p1 = l;
    }
    inline void setB(int l) {
        val.p2 = l;
    }

    inline us toUShort(const float& _f) const {
        float ftrans = (_f + 4.f) * (65536.f / 8.f);
        return us( (_f >= 4.f) ? 65535 : ((_f < -4.f) ? 0 : ftrans));
    }

    operator float() const { return raw; }

    code_t() {}
    code_t(float f) {raw = f;}

    code_t(ui8 a, ui8 b, float l) {
        val.p1 = a;
        val.p2 = b;
        val.l = toUShort(l);
    }

} code_t;




// source: http://stackoverflow.com/questions/23030267/custom-sorting-a-vector-of-tuples
// Functor to compare by the Mth element
template<int M, template<typename> class F = std::less>
struct TupleCompare {
    template<typename T>
    bool operator()(T const &t1, T const &t2) {
        return F<typename std::tuple_element<M, T>::type>()(std::get<M>(t1), std::get<M>(t2));
    }
};

// good old vector calculus
/* given a triangle
 *
 *            C
 *          / |    \
 *         /  |     \
 *      b /   /d     \ a
 *       /   |        \
 *      A -----X-------- B
 *         lc    (1-l)c
 *
 *   we want to calc d(C,X) given squared d(C,B), d(C,A) and l
 *
 *   using |x| := ||x||
 *
 *   |d|^2 = |b- l*c|^2
 *         = |b|^2 + l^2 * |c|^2 - 2 <b, l*c>
 *         = |b|^2 + l^2 * |c|^2 - 2*|b|*l*|c| * cos \alpha
 *
 *   cosinus         |a|^2   = |b|^2 + |c|^2 - 2 |b|*|c|* cos \alpha
 *
 *                              |a|^2 - |b|^2 - |c|^2
 *            =>  cos \alpha =  -----------------------
 *                              -2 * |b| * |c|
 *
 *         = |b|^2 + l^2 * |c|^2 + l * ( |a|^2 - |b|^2 - |c|^2  )
 */
inline float extractDistance(float a, float b, float c, float l) {
    // "a" = ||a||^2, "b" = ||b||^2, "c" = ||c||^2, "l" = \lambda
    // float ll = (l<0) ? 1-l : l;
    return b + l * l * c + l * (a - b - c);
}

template<typename T>
inline bool sanitycheckTriangle(T a, T b, T c) {
    return (sqrt(a) + sqrt(b) + 0.00001 >= sqrt(c)) && (sqrt(a) + sqrt(c) + 0.00001 >= sqrt(b)) && (sqrt(b) + sqrt(c) + 0.00001 >= sqrt(a));
}

/* given a triangle
 *
 *            C
 *          / |    \
 *         /  |     \
 *      b /   /d     \ a
 *       /   |        \
 *      A -----X-------- B
 *         lc    (1-l)c
 *
 *   we want to calc \lambda given squared d(C,B), d(C,A), d(A,B)
 *
 *   cosinus         |a|^2   = |b|^2 + |c|^2 - 2 |b|*|c|* cos \alpha
 *
 *                              |a|^2 - |b|^2 - |c|^2
 *            =>  cos \alpha =  -----------------------
 *                              -2 * |b| * |c|
 *
 *   again:
 *         |l*c| = |b| cos(alpha)
 *
 *         |a|^2 = |b|^2 + |c|^2 - 2 |b| |c| cos(alpha)
 *               = |b|^2 + |c|^2 - 2l |c|^2
 *    =>  -2l    = (|a|^2-|b|^2-|c|^2) / |c|^2
 */
template<typename T>
inline T calcRatio(T a, T b, T c) {
    // "a" = ||a||^2, "b" = ||b||^2, "c" = ||c||^2
    return -0.5f * (a - b - c) / c;
}

#define parallel_for \
  _Pragma(" omp parallel for") \
  for



/* convert element-wise from D to T */
template<typename D = uint8_t, typename T = float>
T * batchcast(D *ptr, uint len ) {

    T* ptr2 = new T[len];
    for (uint l = 0; l < len; ++l) {
        ptr2[l] = static_cast<T>(ptr[l]);
    }
    return ptr2;
}

template<typename T = float>
T * batchcast(T *ptr, uint len ) {
    UNUSED(len);
    return ptr;
}

template<typename D = uint8_t, typename T = float>
void batchcast(D *ptr, T *ptr2, uint len ) {

    for (uint l = 0; l < len; ++l) {
        ptr2[l] = static_cast<T>(ptr[l]);
    }
}


template<typename T> inline void serializeValue(std::fstream &f, T y) {
    f.write(reinterpret_cast<const char*>(&y), sizeof(y));
}
template<typename T> inline void serializeValue(std::fstream &f, T y, uint size) {
    f.write(reinterpret_cast<const char*>(y), size * sizeof(T));
}
template<typename T> inline void unserializeValue(std::fstream &f, T &y) {
    f.read((char*) &y, sizeof(T));
}
template<typename T> inline void unserializeValue(std::fstream &f, T y, uint size) {
    f.read((char*) y,  size * sizeof(T));
}


#endif