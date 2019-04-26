#ifndef LOSS_H
#define LOSS_H

#include <cstdio>
#include <cmath>
#include <stdlib.h>     
#include "tensor.h"



class LossFunction {
private:

public:
    LossFunction();

    float TrainingAccuracy(tensor* networkOutput, tensor* labels);
    tensor* calculate(tensor* networkOutput, tensor* labels, tensor* output);
    float TestAccuracy(tensor* OutputVector, tensor* labels);
};

#endif  /* !LOSS_H */
