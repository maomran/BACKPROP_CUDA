#include <dense.h>

Dense::Dense(int inSize, int outSize){
	this->inSize = inSize;
	this->outSize = outSize;

	 float** initialWeigths = new float*[outSize];
    *initialWeigths = new float[inSize * outSize];
    for (int i = 1; i < outSize; i++) 
    	initialWeigths[i] = initialWeigths[i-1] + inSize;

    // Fill weights with some float numbers
    float minWeight = -1.0f * sqrt(2/inSize);
    float maxWeight = 1.0f * sqrt(2/inSize);
    for (int y = 0; y < outSize; y++) {
        for (int x = 0; x < inSize; x++) {
            initialWeigths[y][x] = randomFloat(minWeight, maxWeight);
        }
    }
    this->weights = new tensor(outSize, inSize, initialWeigths);
    this->deltaWeights = NULL;

    // Prepare place for initial b on CPU
    float* initialBias = new float[outSize];
    
    // Fill weights with some float numbers
    for (int i = 0; i < outSize; i++) {
        initialBias[i] = 0;
    }
    this->b = new tensor(outSize, initialBias);
    this->b_geadients = NULL;

    // Prepare outSize for forward and backprop
    this->outputForward = NULL;
    this->outputBackward = NULL;

    // Clean memory
    delete[] initialWeigths;
    delete[] initialBias;


}