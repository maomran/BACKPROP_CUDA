#ifndef MODEL_H
#define MODEL_H

#include <vector>

#include "layer.h"
#include "funobj.h"
#include "sgd.h"
#include "tensor.h"
// #include "utils.h"


class Model {
public:
    SGD* optimizer;
    CrossEntropyLoss* funobj;
    std::vector<Layer*> layers;
    tensor* gradients;

    Model(SGD* optimizer, CrossEntropyLoss* funobj);

    void addLayer(Layer* layer);
    tensor* forward(tensor* input);
    void backward(tensor* output, tensor* layers);
};

#endif  /* MODEL_H */
