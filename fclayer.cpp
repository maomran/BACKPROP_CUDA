#include <fclayer.h>

FCLaier::FCLaier(int inSize, int outSize){
    this->inSize = inSize;
    this->outSize = outSize;
    float minWeight = -1.0f * sqrt(2/inSize);
    float majWeight = 1.0f * sqrt(2/inSize);
    float** initialWeigths = new float*[outSize];
    *initialWeigths = new float[inSize * outSize];

    for (int i = 0; i < outSize; i++) {
        for (int j = 0; j < inSize; j++) {
            initialWeigths[i][j] = randomFloat(minWeight, majWeight);
        }
    }
    float* initialBias = new float[outSize];
    
    // Fill w with some float numbers
    for (int i = 0; i < outSize; i++) {
        initialBias[i] = 0;
    }

    this->w = new tensor(outSize, inSize, initialWeigths);
    this->w_gradient = NULL;
    this->bias = new tensor(outSize, initialBias);
    this->b_geadients = NULL;
    this->outputForward = NULL;
    this->outputBackward = NULL;
}
// TODO: Forward doesn't seem  good
tensor* FCLayer::forward(tensor* data) {
    // Save this data - will be needed for backpropagation
    this->inputData = data;
    if (!this->outputForward) {
        this->outputForward = new tensor(this->w->row, this->inputData->col);
    }

    // Calculate on GPU: Y = x * W + b
    this->inputData->multiply(this->w, this->outputForward);
    this->outputForward->add(this->bias);

    VERBOSE_PRINT("=== Layer %d ===\n", this);
    VERBOSE_PRINT("Input Data = X: %d Y: %d\n", this->inputData->row, this->inputData->col);
    VERBOSE_PRINT("Weights = X: %d Y: %d\n", this->w->row, this->w->col);
    VERBOSE_PRINT("Bias = X: %d\n", this->bias->row);
    VERBOSE_PRINT("Output = X: %d Y: %d\n", this->outputForward->row, this->outputForward->col);
    return this->outputForward;
}
// TODO: Backward doesn't seem  good
tensor* FCLayer::backward(tensor* gradient) {
    if (!this->w_gradient) {
        this->w_gradient = new tensor(gradient->row, this->inputData->row);
    }
    if (!this->b_gradient) {
        this->b_gradient = new tensor(gradient->row);
    }
    this->inputData->multiply(gradient, this->w_gradient);
    gradient->meanX(this->b_gradient);

    VERBOSE_PRINT("\n=== Layer %d ===\n", this);
    VERBOSE_PRINT("Input data = X: %d Y: %d\n", this->inputData->row, this->inputData->col);
    VERBOSE_PRINT("Gradients = X: %d Y: %d\n", gradient->row, gradient->col);
    VERBOSE_PRINT("Weights = X: %d Y: %d\n", this->w->row, this->w->col);
    VERBOSE_PRINT("Delta Weights (%d) = X: %d Y: %d\n", this->w_gradient, this->w_gradient->row, this->w_gradient->col);
    VERBOSE_PRINT("Bias = X: %d\n", this->bias->row);
    VERBOSE_PRINT("Delta Bias (%d) = X: %d\n", this->b_gradient, this->b_gradient->row);

    if (!this->outputBackward) {
        this->outputBackward = new tensor(this->w->col, gradient->col);
    }
    gradient->multiply(this->w, this->outputBackward);
    VERBOSE_PRINT("Output = X: %d Y: %d\n", this->outputBackward->row, this->outputBackward->col);
    return this->outputBackward;
}
