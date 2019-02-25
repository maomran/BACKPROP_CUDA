#include "layer.h"

tensor* Layer::getWeights() {
    return this->weights;
}

tensor* Layer::getBias() {
    return this->bias;
}

tensor* Layer::getDeltaWeights() {
    return this->deltaWeights;
}

tensor* Layer::getDeltaBias() {
    return this->deltaBias;
}
