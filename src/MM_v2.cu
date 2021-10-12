#include "cuda_runtime.h"
#include <iostream>
#include <stdlib.h>

using namespace std;

int save_parameter(const char* filename, int size, float *parameter);
float* get_parameter(const char* filename, int size);
__global__ void MM_v2(float*,float*,float*);

int main(){
    const char M1[] = "../data/Matrix1.bin";
    const char M2[] = "../data/Matrix2Trans.bin";
    const char Out[]="../data/MM_v2_Result.bin";

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

    for(int i=0;i<20;i++){
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

        MM_v2<<<dim3(128,1),dim3(16,16)>>>(d_A,d_B,d_C);
        cudaDeviceSynchronize();

        cudaEventRecord(stop1,NULL);
        cudaEventSynchronize(start1);
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

__global__ void MM_v2(float* d_A, float* d_B, float* d_C){
    int Batch = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float output[8][8]={0};
    for(int k=0;k<64;k++){
        for(int i=0;i<128;i++){
            output[k/8][k%8] += d_A[Batch*128*128+tx*8*128 + k/8*128 +i]*d_B[Batch*128*128+ty*8*128 + k%8*128 +i];
        }
        d_C[Batch*128*128+tx*8*128 + k/8*128 + ty*8 + k%8] = output[k/8][k%8];
    }
}

int save_parameter(const char* filename, int size, float *parameter) {
    FILE* ptr = fopen(filename,"wb");

    if(!ptr){
        printf("Bad file path: %p, %s\n", ptr, strerror(errno));
        exit(0);
    }
    int cnt = fwrite(parameter,sizeof(float),size,ptr);
    fclose(ptr);
    return cnt;
}   

float* get_parameter(const char* filename, int size) {
    float* parameter = (float*)malloc(size * 4);
    if (!parameter) {
        printf("Bad Malloc\n");
        exit(0);
    }
    FILE* ptr = fopen(filename, "rb");
  
    if (!ptr) {
        printf("Bad file path: %p, %s\n", ptr, strerror(errno));
        exit(0);
    }
    fread(parameter, size * 4, 1, ptr);
  
    fclose(ptr);
    return parameter;
}
