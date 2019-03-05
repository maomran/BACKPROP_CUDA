#ifndef MNIST_H
#define MNIST_H

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "tensor.h"
// #include "utils.h"

class MNISTDataSet {

public:
    float** images;
    float** labels;
    int size;
    // TRUE for Training, False for Test
    MNISTDataSet(bool OP = true );
    
    int getSize();
    void shuffle();
    tensor* getBatchOfImages(int index, int size);
    tensor* getBatchOfLabels(int index, int size);
};

#endif  /* MNIST_H */
