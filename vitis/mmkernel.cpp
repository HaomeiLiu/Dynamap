#include <stdio.h>
#include "./src_tmpl.h"

// cnn layer metadata: can be modified as top-kernel args for host control 
// (except Cin - set as largest Cin in CNN)
#define Cin 64
#define Stride 1
#define K_H 1
#define K_W 1
#define Cout 128
#define Inmap_H 16
#define Inmap_W 16
#define Fmapo_H INmap_H/L1_Stride
#define Fmapo_W INmap_W/L1_Stride

// hardware parameters: cannot be changed for different kernel calls
#define Cout_UF 32
#define Fmap_UF 32
// #define Wdepth Cout/Cout_UF*Cin*K_H*K_W

// hyper-params for winograd
#define m 2
#define r 3
#define Inmap_H_wino Inmap_H/m
#define Inmap_W_wino Inmap_W/m

// Fmapo_H*fmapo_W*Cin, Cin*Cout

// ALG_encode: 0-im2col; 1-kn2row; 2-winograd
extern "C" {
    void top_kernel(dtype *inMap,  dtype *knl, dtype* outMap, ALG_encode alg_last, ALG_encode alg_current,ALG_encode alg_next) {              

        #pragma HLS INTERFACE m_axi port=inMap bundle=gmem0 offset=slave
        #pragma HLS INTERFACE s_axilite port=inMap bundle=control
        #pragma HLS INTERFACE m_axi port=knl bundle=gmem1 offset=slave
        #pragma HLS INTERFACE s_axilite port=knl bundle=control
        #pragma HLS INTERFACE m_axi port=outMap bundle=gmem2 offset=slave
        #pragma HLS INTERFACE s_axilite port=outMap bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control


        hls::stream<blockvec<Fmap_UF>>   L1_inmap;
        hls::stream<blockvec<Cout_UF>>  L1_omap;
        // hls::stream<blockvec<Cin>>   skpMap_s;
        static blockvec<Cout_UF> knl_ram[Cin];
        #pragma HLS ARRAY_PARTITION variable=knl_ram dim=1 complete
        // dtype bneck1_knl_ram[Cin][L2_Cin];   // L1_Cout == L2_Cin
        


        // --------------------------------------------------------------------------------- //
        // Task 0:
        // load input feature map.
        // No matter what alg_last is, their output is always 3D tensor - Cin*INmap_H*INmap_W
        // alg_current = im2col: 3D tensor -> Toeplitz
        // alg_current = kn2row 
        // alg_current = winograd: compute 
        int AccDim;
        int round_MM1;
        int round_MM2;

        #pragma HLS DATAFLOW

        if (alg_current==0):
        {
        	round_MM1=Fmapo_H*Fmapo_W/Fmap_UF * (Cout/Cout_UF); //tiles
        	round_MM2=1;
        	AccDim = Cin*K_H*K_W;
        	// loadIn for im2col
	        // input feature map -> img2col matrix
	        // no trick here cuz it's just for ordinary convolution
        	streamInMap<Fmap_UF, Cin, Fmapo_H, Fmapo_W, K_H, K_W>(inMap, L1_inmap);
        	// loadW for im2col
        	loadW<Cin, Cout, Cout_UF, K_H, K_W>(knl, knl_ram);
        }

        if (alg_current==1): //kn2row needs to be called k^2 times
        {
        	round_MM1=Fmapo_H*Fmapo_W/Fmap_UF * (Cout/Cout_UF); //tiles
        	round_MM2=K_H*K_W; //pad-acc, outer
        	AccDim = Cin;
        	// loadW for kn2row
        }


        if (alg_current==2): //wino needs to be called (m+r-1)*(m+r-1) times, control from host?
        {
        	int round_MM=(m+r-1)*(m+r-1);
        	AccDim = Cin;
        	// loadW for wino

        	// tranform W for wino
        	void TransformW_wino();
        	// loadIn for wino
        	
        	// tranform Inmap for wino
        	void TransformIn_wino();
        }
        

        // Task 1:
        // perform conv
	    for (int i2 = 0; i2 < round_MM2; i2++) {
	    	for (int i1 = 0; i1 < round_MM1; i1++) {
	    		matmulcore<Fmap_UF, Cout_UF, AccDim>(L1_inmap, knl_ram, L1_omap);
	    		if (alg_current==1):
	    			// note: change this to support different values of K_H,k_W, o_h, o_w. also chaneg to support
	    			// cout aggregated output layout
	    			// itr1 should be i2/Fmap_UF, itr2 should be i2%Fmap_UF?
	    			padacc<Stride, K_H, Fmapo_H, 1, 1, i1>(hls::stream<blockvec_Out_P> Inrows[n],hls::stream<blockvec_Out_P> &outpipe)
	    	}
	    }

        // Task 2:
        // store result back to RAM
        // store<L3_Cout,Fmap_H,Fmap_W>(outMap, rslt_s);
    	if (alg_current==0 or alg_current==1){
    		storeDDR<int Cout, int Cout_UF, int AccDim, int Fmapo_H, int Fmapo_W>(outMap, L1_omap);
    	}
    	if (alg_current==2){
			TransformOut_wino(/*...*/);
			storeDDR_wino(/*...*/);
    	}
        // ------------------------------------------------------------------------- //

    }
}

// Usually, Aggf = Fmap_UF
template<int Aggf, int Cin, int Height, int Width, int K_H, int K_W>
void streamInMap(const dtype inMap[], hls::stream<blockvec<Cout>> outMap) {
    for (int h = 0; h < Height; h++) {
    	for (int w = 0; w < Width; w++) {
    		for (int k1 = 0; k1 < K_H; k1++){
    			for (int k2 = 0; k2 < K_W; k2++){
		    		for (int p2 = 0; p2 < Cin/Aggf; p2++) {
		    			#pragma HLS PIPELINE 
				        map_blockvec<Aggf> tempA;
				        #pragma HLS aggregate variable=tempA
				        loadIn_3 : for (int pf = 0; pf < Aggf; pf++) {
				              
				            tempA.d[pf] = inMap[((h+k1)*Inmap_W+w+k2)*Cin+p2*Aggf+pf];
				        }
				        outMap.write(tempA);
		    		}    				
    			}
    		}
		}
    }
}


// Assume W bvs in col major order - K2, Cin, Cout
template<int Cin, int Cout, int Cout_UF, int K_H, int K_W>
void loadW(const dtype W[], blockvec<Cout_UF> Wcols[]){
	#pragma HLS aggregate variable=W
	#pragma HLS aggregate variable=Wcols

	for (int k=0; k<K_H*K_W; k++){
		for (int i=0; i<Cin; i++){
			for (int o=0; o<Cout/Cout_UF; o++){
				#pragma HLS PIPELINE 
				map_blockvec<Cout_UF> tempA;
				#pragma HLS aggregate variable=tempA
				for (int o2=0; o2<Cout_UF; o2++){
					tempA.d[o2] = W[(k*Cin*Cout)+i*Cout+o*Cout_UF+o2];
				}
				Wcols[i].write[tempA];
			}
		}
	}
}

template<int Cout, int Cout_UF, int AccDim, int Fmapo_H, int Fmapo_W>
void storeDDR(dtype C[], hls::stream<blockvec<Cout_UF>> &outpipe){
#pragma HLS aggregate variable=C
	// int hpal=(H/Pa<1)?1:H/Pa;

	for (int j = 0; j < Fmapo_H*Fmapo_W; j++){
		for (int i = 0; i < Cout/Cout_UF; i++){
			for (int ii = 0; ii < Cout_UF; ii++){
				#pragma HLS PIPELINE
				blockvec<Cout_UF> temp=outpipe.read();
				C[j*Cout+i*Cout_UF+ii] = temp[ii];
			}
		}
	}
	
}

void TransformIn_wino(){
	// RAISE NOT IMPLEMENTED
}

void TransformW_wino(){
	// RAISE NOT IMPLEMENTED
}

void TransformOut_wino(){
	// RAISE NOT IMPLEMENTED
}

void storeDDR_wino(dtype C[], hls::stream<blockvec<Cout_UF>> &outpipe /*m,r*/){
	// RAISE NOT IMPLEMENTED
}

//Inrows: co blockvecs (each size Pa)
//Wcols: co wblockvecs (each size Ta)
//Crows: Ta blockvecs (each size Pa)
//input fmap: [o^2,Cout] broadcast
//weights: [Cout,Cin]
template<int Fmap_UF, int Cout_UF, int AccDim>
void matmulcore(hls::stream<blockvec<Fmap_UF>> &Inrows, blockvec<Cout_UF> Wcols[], hls::stream<blockvec<Cout_UF>> &Crows) {
#pragma HLS aggregate variable=Inrows
#pragma HLS aggregate variable=Wcols
#pragma HLS aggregate variable=Crows
	dtype C[Fmap_UF][Cout_UF]; 
	#pragma HLS bind_storage variable=C type=RAM_2P impl=LUTRAM
	#pragma HLS ARRAY_PARTITION variable=C dim=1 complete
	#pragma HLS ARRAY_PARTITION variable=C dim=2 complete

	blockvec<Fmap_UF> tempA;
	blockvec<Cout_UF> tempB;
	int k=0;
   	#pragma HLS aggregate variable=tempA
    #pragma HLS aggregate variable=tempB
	#pragma HLS dependence array variable=C inter false
	// for(int yy = 0; yy < LL*BSIZE/P*LN/T; yy++){
	for(int yy = 0; yy < AccDim; yy++){ //outer i,j,k loops
	#pragma HLS PIPELINE II = 1
		#pragma HLS dependence array variable=C inter false
		k = yy/(1);

		tempA = Inrows.read();
		tempB = Wcols[k];

		for(int ii = 0; ii < Fmap_UF; ii++) { //old P
			#pragma HLS UNROLL
			for(int jj = 0; jj < Cout_UF; jj++) { //old T
				#pragma HLS UNROLL
				#pragma HLS dependence variable=C inter false
				dout_t c1;
				#pragma HLS BIND_OP variable=c1 op=mul impl=dsp latency=2
				c1=tempA.a[ii] * tempB.a[jj];
				int c1_int=c1;
				C[ii][jj] = (k==0)? 0: C[ii][jj] + c1_int;
			}
		}
	}

	//write out to stream
		for(int ii = 0; ii < Fmap_UF; ii++) {
   		#pragma HLS PIPELINE
			blockvec<Cout_UF> tempC;
			#pragma HLS aggregate variable=tempC
				
				for(int jj = 0; jj < Cout_UF; jj++) {
					int tmp_c=C[ii][jj];
					// =(tmp_c>0)?tmp_c:0;
					// tempC.a[i*P+ii]=C[i][j][ii][jj];
					//relu activation implemented

					tempC.a[ii]=(tmp_c>0)?tmp_c: 0;
				}
			}
			Crows.write(tempC);
		
	
}


//relu implemented

//pad-acc module
//Inrows: o^2*cin (need to output H^2*cin) use (H)^2 BRAM banks H is largest H in the network? may use uram
//each Inrow stream has Ta bvs
//o^2:81,49,etc
//kernel 0 output(O^2 pixel-bars(each bar size Ta)): bar[ii][jj] goes to scratchpad[K-1+ii*S][K-1+jj*S]
//kernel 1 output(O^2 pixel-bars(each bar size Ta)): bar[ii][jj] goes to scratchpad[K-2+ii*S][K-1+jj*S]
// ...
//kernel y=(a*K+b) {we can index by a,b where a=y/K,b=y%k} 
// kernel y output bar[ii][jj] goes to scratchpad[K-1-a+ii*S][K-1-b+jj*S] 
// a,b ranges:0~K-1
//0<=ii,jj<O, cut by Pa & identified by Pa&it1. ii*O+jj=it1*Pa+kk
template<int S, int K, int O, int it1, int it2, int itk>
void padacc(hls::stream<blockvec_Out_P> Inrows[n],hls::stream<blockvec_Out_P> &outpipe){
#pragma HLS aggregate variable=Inrows
	blockvec_Out_P tempO2buffer[n][Ta];
	blockvec_Out_P tmp[n];
	#pragma HLS ARRAY_PARTITION variable=tempO2buffer dim=0 complete
	#pragma HLS ARRAY_PARTITION variable=tmp dim=0 complete
	#pragma HLS aggregate variable=tempO2buffer
	#pragma HLS aggregate variable=tmp
	int scratchpad[H][H][Ta]; //Cin=some number times Ta
	#pragma HLS bind_storage variable=scratchpad type=RAM_2P impl=bram
 // #pragma HLS RESOURCE variable=scratchpad core=XPM_MEMORY uram
	//#pragma HLS ARRAY_PARTITION variable=scratchpad dim=1 complete
	#pragma HLS ARRAY_PARTITION variable=scratchpad dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=scratchpad dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=tempO2buffer dim=2 complete
	int y=0; //y: which 1x1 kernel output is this, out of all K^2=itk*n
	int yy=0; //yy: which pixel-bar is this, out of all O^2 bars in this 1x1 kernel output. yy->ii,jj
	int a=0;
	int b=0;
	int ii=0;
	int jj=0;
	PadAccLoadLoop:for (int j = 0; j < Ta; j++){
		#pragma HLS PIPELINE
		for (int i = 0; i < n; i++){
			#pragma HLS UNROLL
			tmp[i]=Inrows[i].read();
			tempO2buffer[i][j] = tmp[i];
		}
	}
	PadAccNLoop:for (int i = 0; i < n; i++){ //each 1by1 kernel should find corresponding (different but may overlapping) index in sctratchpad 

		a=(itk*n+i)/K;
		b=(itk*n+i)%K;
		// #pragma HLS PIPELINE
		// #pragma HLS dependence array variable=scratchpad intra false
		PadAccPaLoop: for (int kk = 0; kk < Pa; kk++){ //each output pixel should find corresponding (different) index in sctratchpad
			// int y=itk*8+it1*Pa+kk; //y: which 1x1 kernel is this, out of all K^2=itk*n
			#pragma HLS PIPELINE II=1
			ii=(it1*Pa+kk)/O;
			jj=(it1*Pa+kk)%O;
			// #pragma HLS PIPELINE 
			#pragma HLS dependence variable=scratchpad inter false
			// #pragma HLS dependence array variable=scratchpad intra false
			PadAccUnrollLoop:for (int j = 0; j < Ta; j++){ //Ta->which Ta in Cin
				#pragma HLS dependence array variable=scratchpad inter false
				int ind1=K-1-a+ii*S;
				int ind2=K-1-b+jj*S;
				scratchpad[ind1][ind2][j] = (it1==0 && itk==0) ? 0:scratchpad[ind1][ind2][j]+tempO2buffer[i][j].a[kk];
			}
		}
	}
	if (itk==k_bound-1 && it1==O2/Pa-1){
		// storeDDR(C, scratchpad, 0,it2); //it1 not useful
		outPipeH1Loop:for (int j = 0; j < H; j++){
			#pragma HLS pipeline II=1
			blockvec_Out_P tmp [Ta*H/Pa]; //total size=(H/Pa)*Pa*Ta=H*Ta
			#pragma HLS ARRAY_PARTITION variable=tmp dim=1 complete
			#pragma HLS bind_storage variable=tmp type=RAM_1P impl=uram
			for (int i = 0; i < Ta; i++){
				for (int k = 0; k < H; k++){
					// #pragma HLS UNROLL
					int i1=k/Pa;
					int i2=k%Pa;
					tmp[i*H/Pa+i1].a[i2]=scratchpad[j][k][i];
				}
			}
			outPipeLoop:for (int i = 0; i < Ta*H/Pa; i++){
				#pragma HLS pipeline II=1
				outpipe.write(tmp[i]);
			}
		}
	}
}





