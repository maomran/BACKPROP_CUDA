#ifndef UTILS_H
#define UTILS_H
#include <string>
#include <cstdlib>
#include <cstdio>


#define 32 32
#define 32 32
#define BIDX 32
#define BIDY 32
#define LOG_FILE_NAME       "log.csv"
#define DEBUG 1
#define LR       1e-06
#define EPOCHS    100
#define BATCHSIZE          512

#if defined(DEBUG) && DEBUG >= 1
 #define VERBOSE_PRINT(fmt, args...) fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##args)
#else
 #define VERBOSE_PRINT(fmt, args...)
#endif


float randomFloat(float a, float b) {
    return a + static_cast <float> (std::rand()) /( static_cast <float> (RAND_MAX/(b-a)));
}

int randomInt(int a, int b) {
    return a + std::rand()%(b-a);
}


#endif