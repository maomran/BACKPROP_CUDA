#ifndef LOSS_H
#define LOSS_H

#include <cstdio>
#include <cmath>

#include "tensor.h"

#define VERY_SMALL_NUMBER 1e-10


class CrossEntropyLoss {
private:

public:
    CrossEntropyLoss();

    tensor* calculate(tensor* networkOutput, tensor* labels, tensor* output);
    float getLoss(tensor* networkOutput, tensor* labels);
    float getAccuracy(tensor* networkOutput, tensor* labels);
};

#endif  /* !LOSS_H */
