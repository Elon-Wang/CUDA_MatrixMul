// CUDA runtime 库 + CUBLAS 库
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <iostream>
#include <stdlib.h>

using namespace std;

int save_parameter(const char* filename, int size, float *parameter);
float* get_parameter(const char* filename, int size);
void Memory_device(float ** &A,float * &A_data,int n,int m);
__global__ void init_matrix(float **A, float *A_data, int n, int m);

int main(){
    const char M1[] = "../data/Matrix1.bin";
    const char M2[] = "../data/Matrix2.bin";
    const char Out[]="../data/CublasResult.bin";

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


    cublasStatus_t ret;
    for(int i=0;i<20;i++){
        // The actual matrix are in 3D, which is (n,h,w).
        // We put each (h,w) in an array, and the n as the index to indentify the matrix
        // For example, the dimension of B (kelnal) is (128,128,128), the first 128 means 
        // 128 different matrix. Each Matrix has a dimension of (128,128), and was put in 
	// an array, preparing for the matrix Multiplication in CuBlasSgemm function.
        float **d_A,*d_A_data;  
        float **d_B,*d_B_data;
        float **d_C,*d_C_data;
        Memory_device(d_A, d_A_data, Batch, A_ROW* A_COL);
        Memory_device(d_B, d_B_data, Batch, B_ROW* B_COL);
        Memory_device(d_C, d_C_data, Batch, A_ROW* B_COL);

        // copy the data from host to the device
        cudaMemcpy(d_A_data,h_A,sizeof(float)*nA,cudaMemcpyHostToDevice); //数据从内存拷贝到显存
        cudaMemcpy(d_B_data,h_B,sizeof(float)*nB,cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1, beta = 0;
        
        cudaEvent_t start1,stop1;
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);
        cudaEventRecord(start1,NULL);
        ret = cublasSgemmBatched(
                handle,
                CUBLAS_OP_N,   //矩阵A的属性参数，不转置，按列优先
                CUBLAS_OP_N,   //矩阵B的属性参数，不转置，按列优先
                B_COL,         //矩阵B^T、C^T的行数
                A_ROW,         //矩阵A^T、C^T的列数
                B_ROW,         //B^T的列数，A^T的行数，此处也可为A_COL,一样的
                &alpha,        //alpha的值
                d_B,           //左矩阵，为B^T
                B_COL,         //B^T的leading dimension，按列优先，则leading dimension为B^T的行数(B的列数)
                d_A,           //右矩阵，为A^T
                A_COL,         //A^T的leading dimension，按列优先，则leading dimension为A^T的行数(A的列数)
                &beta,         //beta的值
                d_C,           //结果矩阵C
                B_COL,         //C^T的leading dimension，C^T矩阵一定按列优先，则leading dimension为C^T的行数(C的列数)
                Batch
        );
        cudaEventRecord(stop1,NULL);
        cudaEventSynchronize(start1);
        cudaEventSynchronize(stop1);
        
        // measure and print out the time
        float msecTotal1 = 0.0f;
        cudaEventElapsedTime(&msecTotal1,start1,stop1);
        cudaEventDestroy(start1);
        cudaEventDestroy(stop1);
        printf("time is %lf us\n",msecTotal1*1000);

		/*
        if (ret == CUBLAS_STATUS_SUCCESS)
            printf("sgemm success  %d, line(%d)\n", ret, __LINE__);
        */

        cudaMemcpy(h_C,d_C_data,sizeof(float)*nC,cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_A_data);
        cudaFree(d_B_data);
        cudaFree(d_C_data);
    }
    int cnt = save_parameter(Out,nC, h_C) * sizeof(float);
    printf("%d bytes have been writen to the file %s\n",cnt,Out);

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
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

__global__ void init_matrix(float **A, float *A_data, int n, int m){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid<n){
        A[tid] = &A_data[tid*m];
    }
}
 
void Memory_device(float ** &A,float * &A_data,int n,int m){
    cudaMalloc((void **)&A_data,sizeof(float)*n*m);
    cudaMalloc((void **)&A, n * sizeof(float*));
 
    init_matrix<<<1,n>>>(A,A_data,n,m);
}
