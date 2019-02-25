#ifndef SGD_H
#define SGD_H

#include <cstdio>
#include <cmath>

#include "layer.h"
#include "tensor.h"


class SGD: public {

public:
    float lr;
    SGD(float lr);
    void optimize(FCLayer* layer);
};

#endif  /* !SGD_H */
