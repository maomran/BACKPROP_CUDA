#include "model.h"

Model::Model(SGD* optimizer, LossFunction* funobj) {
    this->optimizer = optimizer;
    this->funobj = funobj;
    this->gradients = NULL;
}

void Model::addLayer(Layer* layer) {
    this->layers.push_back(layer);
}

tensor* Model::forward(tensor* input) {
    tensor* values = input;

    for (vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); layer++) {
        values = (*layer)->forward(values);
    }
    return values;
}

void Model::backward(tensor* output, tensor* labels) {
    if (!this->gradients) {
        this->gradients = new tensor(output->row, output->col);
    }
    this->funobj->calculate(output, labels, this->gradients);

    tensor* values = gradients;
    for (vector<Layer*>::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); layer++) {
        values = (*layer)->backward(values);
    }

    for (vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); layer++) {
        optimizer->optimize(*layer);
    }
}
