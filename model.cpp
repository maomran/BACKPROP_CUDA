#include "model.h"

Model::Model(SGD* optimizer, CrossEntropyLoss* funobj) {
    this->optimizer = optimizer;
    this->funobj = funobj;
    this->gradients = NULL;
}

void Model::addLayer(Layer* layer) {
    VERBOSE_PRINT("Adding Layer to the model: %d\n", layer);
    this->layers.push_back(layer);
}

tensor* Model::forward(tensor* input) {
    tensor* values = input;
    for (std::vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); layer++) {
        values = (*layer)->forward(values);
        #if defined(DEBUG) && DEBUG >= 2
        VERBOSE_PRINT("Forward pass for Layer %d:\n", (*layer));
        values->toString();
        #endif
    }
    return values;
}

void Model::backward(tensor* output, tensor* labels) {
    // Compute gradients with funobj function
    if (!this->gradients) {
        this->gradients = new tensor(output->getSize(X), output->getSize(Y));
    }
    this->funobj->calculate(output, labels, this->gradients);
    #if defined(DEBUG) && DEBUG >= 2
    VERBOSE_PRINT("Backward pass gradients:\n");
    gradients->toString();
    #endif

    // Pass these gradients with backpropagation
    tensor* values = gradients;
    for (std::vector<Layer*>::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); layer++) {
        values = (*layer)->backward(values);
        #if defined(DEBUG) && DEBUG >= 2
        VERBOSE_PRINT("\nBackward pass for Layer %d:\n", (*layer));
        values->toString();
        #endif
    }

    // Updates all layers with optimizer
    for (std::vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); layer++) {
        optimizer->optimize(*layer);
    }
}
