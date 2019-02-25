#include "sgd.h"

SGD::SGD(float lr) {
    this->lr = lr;
}

void SGD::optimize(Layer* layer) {
    if (layer->w_gradients != NULL) {
        layer->w_gradients->scale(this->lr);
    }
    if (layer->b_gradients != NULL) {
        layer->b_gradients->scale(this->lr);
    }

    if (layer->w != NULL) {
        layer->w->subtract(layer->w_gradients);
    }
    if (layer->bias != NULL) {
        layer->bias->subtract(layer->b_gradients);
    }
}
