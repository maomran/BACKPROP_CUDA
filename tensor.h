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

    float* DevData();
    float** Dev2Host();
    
    void add(tensor* tensor_t);
    tensor* subtract(tensor* tensor_t, tensor* output);
    void scale(float factor);
    tensor* multiply(tensor* t, tensor* output);
    tensor* avg(tensor* output);
    void toString();
	
};
#endif