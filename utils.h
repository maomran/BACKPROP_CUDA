##ifndef UTILS_H
#define UTILS_H
#include <string>
#include <cstdlib>
#include <cstdio>


#define TIDX 32
#define TIDY 32
#define BIDX 32
#define BIDY 32
#define LOG_FILE_NAME       "logs/experiment.csv"
#define DEBUG 1

#if defined(DEBUG) && DEBUG >= 1
 #define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##args)
#else
 #define DEBUG_PRINT(fmt, args...)
#endif


float randomFloat(float a, float b) {
    return a + static_cast <float> (std::rand()) /( static_cast <float> (RAND_MAX/(b-a)));
}

int randomInt(int a, int b) {
    return a + std::rand()%(b-a);
}


#endif