#include "sgd.h"

SGD::SGD(float lr) {
    this->lr = lr;
}

void SGD::optimize(Layer* layer) {
    if (layer->w_gradient != NULL) {
        layer->w_gradient->Scale(this->lr);
    }
    if (layer->b_gradient != NULL) {
        layer->b_gradient->Scale(this->lr);
    }

    if (layer->w != NULL) {
        layer->w->Subtract(layer->w_gradient);
    }
    if (layer->bias != NULL) {
        layer->bias->Subtract(layer->b_gradient);
    }
}
