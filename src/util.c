#include "util.h"

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