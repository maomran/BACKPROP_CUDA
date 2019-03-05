#include "sgd.h"

SGD::SGD(float lr) {
    this->lr = lr;
}

void SGD::optimize(Layer* layer) {
    if (layer->w_gradient != NULL) {
        layer->w_gradient->scale(this->lr);
    }
    if (layer->b_gradient != NULL) {
        layer->b_gradient->scale(this->lr);
    }

    if (layer->w != NULL) {
        layer->w->subtract(layer->w_gradient);
    }
    if (layer->bias != NULL) {
        layer->bias->subtract(layer->b_gradient);
    }
}
