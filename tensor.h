#ifndef TENSOR_H
#define TENSOR_H

#include <cstdio>

class tensor {
public:
	int row;
	int col;
	float *d_data;

	tensor(int row, int col);
	tensor(int row, int col, float *d_data);
	tensor(int row, int col, float **h_data);
    ~tensor();
    float** Dev2Host();
    
    void Add(tensor* input);
    void Subtract(tensor* input);
    void Scale(float factor);
    tensor* MatMul(tensor* t, tensor* output);
    tensor* GradientMul(tensor* t, tensor* output);
    tensor* BackwardMul(tensor* t, tensor* output);
    tensor* GradAvg(tensor* output);
    void toString();
	
};
#endif