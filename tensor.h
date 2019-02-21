#ifndef TENSOR_H
#define TENSOR_H

#include <cstdio>

struct tensor
{
	
};
class tensor
{
public:
	int row;
	int col;
	float *data;

	tensor(int row, int col);
	tensor(int row, int col, float *d_data);
	tensor(int row, int col, float **h_data);
	~tensor();

    float* Host2Dev();
    float** Dev2Host();
    
    tensor* add(tensor* t);
    tensor* subtract(tensor* t);
    void scale(float factor);
    tensor* multiply(tensor* t, tensor* output);
    tensor* multiplyByTransposition(tensor* t, tensor* output);
    tensor* transposeAndMultiply(tensor* t, tensor* output);
    tensor* avg(tensor* output);

	
};
#endif