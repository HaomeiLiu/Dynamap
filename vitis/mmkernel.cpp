#include <stdio.h>
#include "./mmkernel.h"

// cnn layer metadata: can be modified as top-kernel args for host control 
// (except Cin - set as largest Cin in CNN)
#define CIN 64
#define STRIDE 1
#define K_H 1
#define K_W 1
#define K_BOUND K_H*K_W
#define COUT 128
#define INMAP_H 16
#define INMAP_W 16
#define FMAPO_H INMAP_H/STRIDE
#define FMAPO_W INMAP_W/STRIDE
#define O_2 FMAPO_H*FMAPO_W

// hardware parameters: cannot be changed for different kernel calls
#define COUT_UF 32
#define FMAP_UF 32
// #define Wdepth Cout/Cout_UF*Cin*K_H*K_W

// hyper-params for winograd
#define m 2
#define r 3
#define Inmap_H_wino INMAP_H/m
#define Inmap_W_wino INMAP_W/m

// Fmapo_H*fmapo_W*Cin, Cin*Cout

// ALG_encode: 0-im2col; 1-kn2row; 2-winograd
//inMap - horizontal
//knl_ram - vertical 
extern "C" {
    void top_kernel(dtype *inMap,  dtype *knl, dtype* outMap, ALG_encode alg_last, ALG_encode alg_current,ALG_encode alg_next) {              

        #pragma HLS INTERFACE m_axi port=inMap bundle=gmem0 offset=slave
        #pragma HLS INTERFACE s_axilite port=inMap bundle=control
        #pragma HLS INTERFACE m_axi port=knl bundle=gmem1 offset=slave
        #pragma HLS INTERFACE s_axilite port=knl bundle=control
        #pragma HLS INTERFACE m_axi port=outMap bundle=gmem2 offset=slave
        #pragma HLS INTERFACE s_axilite port=outMap bundle=control

		#pragma HLS INTERFACE s_axilite port=alg_last bundle=control
		#pragma HLS INTERFACE s_axilite port=alg_current bundle=control
		#pragma HLS INTERFACE s_axilite port=alg_next bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control


        hls::stream<blockvec<FMAP_UF>>   L1_inmap;
        hls::stream<blockvec<COUT_UF>>  L1_omap;
        // hls::stream<blockvec<Cin>>   skpMap_s;
        blockvec<COUT_UF> knl_ram[CIN];
        #pragma HLS ARRAY_PARTITION variable=knl_ram dim=1 complete //Partitions knl_ram into smaller individual elements.

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

        if (alg_current==0)
        {
        	round_MM1=FMAPO_H*FMAPO_W/FMAP_UF * (COUT/COUT_UF); //tiles
        	round_MM2=1;
        	AccDim = CIN*K_H*K_W;
        	// loadIn for im2col
	        // input feature map -> img2col matrix
	        // no trick here cuz it's just for ordinary convolution
        	streamInMap<FMAP_UF, CIN, FMAPO_H, FMAPO_W, K_H, K_W>(inMap, L1_inmap);
        	// loadW for im2col
        	loadW<CIN, COUT, COUT_UF, K_H, K_W>(knl, knl_ram);
        }

        if (alg_current==1) //kn2row needs to be called k^2 times
        {
        	round_MM1=FMAPO_H*FMAPO_W/FMAP_UF * (COUT/COUT_UF); //tiles
        	round_MM2=K_H*K_W; //pad-acc, outer
        	AccDim = CIN;
        	// loadW for kn2row
        }


        if (alg_current==2) //wino needs to be called (m+r-1)*(m+r-1) times, control from host?
        {
        	int round_MM=(m+r-1)*(m+r-1);
        	AccDim = CIN;
        	// // loadW for wino

        	// // tranform W for wino
        	// void TransformW_wino();
        	// // loadIn for wino
        	
        	// // tranform Inmap for wino
        	// void TransformIn_wino();
        }
        

        // Task 1:
        // perform conv
	    for (int i2 = 0; i2 < round_MM2; i2++) {
	    	for (int i1 = 0; i1 < round_MM1; i1++) {
	    		matmulcore<FMAP_UF, COUT_UF>(L1_inmap, knl_ram, L1_omap, AccDim);
	    		if (alg_current==1){
	    			// note: change this to support different values of K_H,k_W, o_h, o_w. also chaneg to support
	    			// cout aggregated output layout
	    			// itr1 should be i2/Fmap_UF, itr2 should be i2%Fmap_UF?
	    			padacc<COUT_UF, STRIDE, K_H, FMAPO_H, 1, 1>(L1_omap,L1_omap,i1);
				}
	    	}
	    }

        // Task 2:
        // store result back to RAM
        // store<L3_Cout,Fmap_H,Fmap_W>(outMap, rslt_s);
    	if (alg_current==0 || alg_current==1){
    		storeDDR<COUT, COUT_UF, FMAPO_H, FMAPO_W>(outMap, L1_omap);
    	}
    	if (alg_current==2){
			// TransformOut_wino(/*...*/);
			// storeDDR_wino(/*...*/);
    	}
        // ------------------------------------------------------------------------- //

    }
}

// Usually, Aggf = Fmap_UF
template<int Aggf, int Cin, int Height, int Width, int K_H_t, int K_W_t>
void streamInMap(const dtype inMap[], hls::stream<blockvec<Aggf>> &outMap) {
    for (int h = 0; h < Height; h++) {
    	for (int w = 0; w < Width; w++) {
    		for (int k1 = 0; k1 < K_H_t; k1++){
    			for (int k2 = 0; k2 < K_W_t; k2++){
		    		for (int p2 = 0; p2 < Cin/Aggf; p2++) {
		    			#pragma HLS PIPELINE 
				        blockvec<Aggf> tempA;
				        #pragma HLS aggregate variable=tempA
				        for (int pf = 0; pf < Aggf; pf++) {
				              
				            tempA.d[pf] = inMap[((h+k1)*INMAP_W+w+k2)*Cin+p2*Aggf+pf];
				        }
				        outMap.write(tempA);
		    		}    				
    			}
    		}
		}
    }
}


// Assume W bvs in col major order - K2, Cin, Cout
template<int Cin, int Cout, int Cout_UF, int K_H_t, int K_W_t>
void loadW(const dtype W[], blockvec<Cout_UF> Wcols[]){
	#pragma HLS aggregate variable=W
	#pragma HLS aggregate variable=Wcols

	for (int k=0; k<K_H_t*K_W_t; k++){
		for (int i=0; i<Cin; i++){
			for (int o=0; o<Cout/Cout_UF; o++){
				#pragma HLS PIPELINE 
				blockvec<Cout_UF> tempA;
				#pragma HLS aggregate variable=tempA
				for (int o2=0; o2<Cout_UF; o2++){
					tempA.d[o2] = W[(k*Cin*Cout)+i*Cout+o*Cout_UF+o2];
				}
				//ERROR
				Wcols[i]=tempA;
			}
		}
	}
}

//ERROR
template<int Cout, int Cout_UF, int Fmapo_H, int Fmapo_W>
void storeDDR(dtype C[], hls::stream<blockvec<Cout_UF>> &outpipe){
#pragma HLS aggregate variable=C
	// int hpal=(H/Pa<1)?1:H/Pa;

	for (int j = 0; j < Fmapo_H*Fmapo_W; j++){
		for (int i = 0; i < Cout/Cout_UF; i++){
			for (int ii = 0; ii < Cout_UF; ii++){
				#pragma HLS PIPELINE
				blockvec<Cout_UF> temp=outpipe.read();
				C[j*Cout+i*Cout_UF+ii] = temp.d[ii];
			}
		}
	}
	
}

// void TransformIn_wino(){
// 	// RAISE NOT IMPLEMENTED
// }

// void TransformW_wino(){
// 	// RAISE NOT IMPLEMENTED
// }

// void TransformOut_wino(){
// 	// RAISE NOT IMPLEMENTED
// }

// void storeDDR_wino(dtype C[], hls::stream<blockvec<Cout_UF>> &outpipe /*m,r*/){
// 	// RAISE NOT IMPLEMENTED
// }

//Inrows: co blockvecs (each size Pa)
//Wcols: co wblockvecs (each size Ta)
//Crows: Ta blockvecs (each size Pa)
//input fmap: [o^2,Cout] broadcast
//weights: [Cout,Cin]
//ERROR
template<int Fmap_UF, int Cout_UF>
void matmulcore(hls::stream<blockvec<FMAP_UF>> &Inrows, blockvec<COUT_UF> Wcols[], hls::stream<blockvec<COUT_UF>> &Crows, int AccDim) {
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
			Crows.write(tempC);
		}		
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
//ERROR
template<int Cout_UF, int S, int K, int O, int it1, int it2>
void padacc(hls::stream<blockvec<COUT_UF>> &Inrows,hls::stream<blockvec<COUT_UF>> &outpipe,int itk){
	int Ta = 16;
	int Pa = 16;
	int n=1;
	int H = 224;
#pragma HLS aggregate variable=Inrows
	blockvec<COUT_UF> tempO2buffer[n][Ta];
	blockvec<COUT_UF> tmp[n];
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
			tmp[i]=Inrows.read();
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
				scratchpad[ind1][ind2][j] = (it1==0 && itk==0) ? 0:scratchpad[ind1][ind2][j]+int(tempO2buffer[i][j].d[kk]);
			}
		}
	}
	if (itk==K_BOUND-1 && it1==O_2/Pa-1){
		// storeDDR(C, scratchpad, 0,it2); //it1 not useful
		outPipeH1Loop:for (int j = 0; j < H; j++){
			#pragma HLS pipeline II=1
			blockvec<COUT_UF> tmp [Ta*H/Pa]; //total size=(H/Pa)*Pa*Ta=H*Ta
			#pragma HLS ARRAY_PARTITION variable=tmp dim=1 complete
			#pragma HLS bind_storage variable=tmp type=RAM_1P impl=uram
			for (int i = 0; i < Ta; i++){
				for (int k = 0; k < H; k++){
					// #pragma HLS UNROLL
					int i1=k/Pa;
					int i2=k%Pa;
					tmp[i*H/Pa+i1].d[i2]=scratchpad[j][k][i];
				}
			}
			outPipeLoop:for (int i = 0; i < Ta*H/Pa; i++){
				#pragma HLS pipeline II=1
				outpipe.write(tmp[i]);
			}
		}
	}
}




