#ifndef SGD_H
#define SGD_H

#include <cstdio>
#include <cmath>

#include "fclayer.h"
#include "tensor.h"


class SGD{

public:
    float lr;
    SGD(float lr);
    void optimize(Layer* layer);
};

#endif  /* !SGD_H */
