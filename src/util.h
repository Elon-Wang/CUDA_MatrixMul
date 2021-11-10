#ifndef __UTIL_H__
#define __UTIL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float* get_parameter(const char* filename, int size);

int save_parameter(const char* filename, int size, float *parameter);


#ifdef __cplusplus
}
#endif

#endif
