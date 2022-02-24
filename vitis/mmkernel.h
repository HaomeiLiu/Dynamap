#include <hls_stream.h>
#include <ap_int.h>

using namespace std;

typedef ap_int<16> dtype;

// model metadata
// const int H = 16;
// const int W = 16;
// const int Ho = 16;
// const int Wo = 16;
// const int Cin = 4;
// const int K1 = 3;
// const int K2 = 3;
// const int Cout = 4;

// const dtype ofst_knl[K1*K1*Cin][2] = {0};
// const dtype norm_knl[K2*K2*Cin][Cout] = {0};

template<int N>
struct blockvec{
    dtype d[N];
};


// top function
extern "C" void top_kernel(dtype *inMap,  dtype *knl, dtype* outMap, ALG_encode alg_last, ALG_encode alg_current,ALG_encode alg_next);


template<int Aggf, int Cin, int Height, int Width, int K_H, int K_W>
void streamInMap(const dtype inMap[], hls::stream<blockvec<Cout>> outMap);

template<int Cin, int Cout, int Cout_UF, int K_H, int K_W>
void loadW(const dtype W[], blockvec<Cout_UF> Wcols[]);

template<int Cout, int Cout_UF, int AccDim, int Fmapo_H, int Fmapo_W>
void storeDDR(dtype C[], hls::stream<blockvec<Cout_UF>> &outpipe);

template<int Fmap_UF, int Cout_UF, int AccDim>
void matmulcore(hls::stream<blockvec<Fmap_UF>> &Inrows, blockvec<Cout_UF> Wcols[], hls::stream<blockvec<Cout_UF>> &Crows);

template<int S, int K, int O, int it1, int it2, int itk>
void padacc(hls::stream<blockvec_Out_P> Inrows[n],hls::stream<blockvec_Out_P> &outpipe);

void TransformIn_wino();

void TransformW_wino();

void TransformOut_wino();

void storeDDR_wino(dtype C[], hls::stream<blockvec<m+r-1>> &outpipe /*m,r*/);
