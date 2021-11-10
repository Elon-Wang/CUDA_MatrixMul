#include "cuda_runtime.h"
#include "util.h"

using namespace std;

__global__ void MM_v5(float*,float*,float*);

int main(){
    const char M1[] = "../data/Matrix1.bin";
    const char M2[] = "../data/Matrix2Trans.bin";
    const char Out[]="../data/MM_v4_Result.bin";

    // the parameter of the two matrix
    // int Batch = 1;
    int Batch = 128;
    int A_ROW = 128;
    int A_COL = 128;
    int B_ROW = 128;
    int B_COL = 128;

    // the number of element of matrix A, B, C 
    int nA = Batch * A_ROW * A_COL;
    int nB = Batch * B_ROW * B_COL;
    int nC = Batch * A_ROW * B_COL;

    // read data from file to the host variables
    float *h_A,*h_B,*h_C;  
    h_A = get_parameter(M1, nA);
    h_B = get_parameter(M2, nB);
    h_C = (float*)malloc(sizeof(float)*nC);

    for(int i=0;i<10;i++){
        float *d_A,*d_B,*d_C;
        cudaMalloc((void **) &d_A, nA<<2);
        cudaMalloc((void **) &d_B, nB<<2);
        cudaMalloc((void **) &d_C, nC<<2);

        cudaMemcpy(d_A, h_A, nA<<2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, nB<<2, cudaMemcpyHostToDevice);
        cudaMemset((void *) d_C, 0, nC<<2);
        
        cudaEvent_t start1,stop1;
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);
        cudaEventRecord(start1,NULL);

        MM_v5<<<dim3(1,128),dim3(256,1)>>>(d_A,d_B,d_C);
        cudaDeviceSynchronize();

        cudaEventRecord(stop1,NULL);
        cudaEventSynchronize(stop1);
        
        // measure and print out the time
        float msecTotal1 = 0.0f;
        cudaEventElapsedTime(&msecTotal1,start1,stop1);
        cudaEventDestroy(start1);
        cudaEventDestroy(stop1);
        printf("time is %lf us\n",msecTotal1*1000);

        cudaMemcpy(h_C, d_C, nC<<2, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    int cnt = save_parameter(Out,nC, h_C) * sizeof(float);
    printf("%d bytes have been writen to the file %s\n",cnt,Out);

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}

__global__ void MM_v5(float* d_A, float* d_B, float* d_C){
    int Batch = blockIdx.y;
    int tx = threadIdx.x;
    int BatchOffset = Batch *128 *128; 

    __shared__ float M1_1[16][65];
    __shared__ float M1_2[16][65];
    __shared__ float M2_1[16][65];
    __shared__ float M2_2[16][65];
    float output[64]={0};
    int idx =tx;
    float reside;

    for (int cnt=0;cnt<8;cnt++){
        
        for(int chn =0;chn<16;chn++){
            if(tx<64){
                idx = tx;
                M1_1[chn][idx] = d_A[BatchOffset + chn * 4 * 128 + idx / 16 *128 +cnt*16 +idx%16];
            }else if(tx<128) {
                idx = tx - 64;
                M1_2[chn][idx] = d_A[BatchOffset + chn * 4 * 128 + idx / 16 *128 +cnt*16 +idx%16 + 64*128 ];
            }else if(tx<192) {
                idx = tx - 128;
                M2_1[chn][idx] = d_B[BatchOffset + cnt*16*128 + chn * 128 +idx ];
            }else {
                idx = tx - 192;
                M2_2[chn][idx] = d_B[BatchOffset + cnt*16*128 + chn * 128 +idx + 64];
            }
        }
        
        // for (int chn = 0;chn<16;chn++){
        //     if(tx<64){
        //         idx = tx;
        //         reside = M2_1[chn][idx];
        //         for(int k = 0;k<64;k++){
        //             output[k] += M1_1[chn][k] * reside; 
        //         } 
        //     }else if(tx<128) {
        //         idx = tx - 64;
        //         reside = M2_2[chn][idx];
        //         for(int k = 0;k<64;k++){
        //             output[k] += M1_1[chn][k] * reside; 
        //         } 
        //     }else if(tx<192) {
        //         idx = tx - 128;
        //         reside = M2_1[chn][idx];
        //         for(int k = 0;k<64;k++){
        //             output[k] += M1_2[chn][k] * reside; 
        //         } 
        //     }else {
        //         idx = tx - 192;
        //         reside = M2_2[chn][idx];
        //         for(int k = 0;k<64;k++){
        //             output[k] += M1_2[chn][k] * reside; 
        //         } 
        //     }
        // }

        for(int k=0;k<64;k++){
            if(tx<64){
                idx = tx;
                for(int chn = 0;chn<16;chn++){
                    output[k] += M1_1[chn][k] * M2_1[chn][idx]; 
                } 
            }else if(tx<128) {
                idx = tx - 64;
                for(int chn = 0;chn<16;chn++){
                    output[k] += M1_1[chn][k] * M2_2[chn][idx]; 
                } 
            }else if(tx<192) {
                idx = tx - 128;
                for(int chn = 0;chn<16;chn++){
                    output[k] += M1_2[chn][k] * M2_1[chn][idx]; 
                } 
            }else {
                idx = tx - 192;
                for(int chn = 0;chn<16;chn++){
                    output[k] += M1_2[chn][k] * M2_2[chn][idx]; 
                } 
            }
        }
    }
    for(int k=0;k<64;k++){
        if(tx<128){
            d_C[BatchOffset + k*128+ tx] = output[k];
        }else{
            d_C[BatchOffset + k*128+ tx+ 64*128 -128] = output[k];
        }
    }
}