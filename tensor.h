#ifndef TENSOR_H
#define TENSOR_H

#include <cstdio>

struct tensor
{
	
};
class tensor
{
public:
	int idx;
	int idy;
	float *data;

	tensor(int idx, int idy);
	tensor(int idx, int idy, float *d_data);
	tensor(int idx, int idy, float **h_data);
	~tensor();

    float* Host2Dev();
    float** Dev2Host();
    
    void add(Tensor2D* tensor);
    void subtract(Tensor2D* tensor);
    void scale(float factor);
    Tensor2D* multiply(Tensor2D* tensor, Tensor2D* output);
    Tensor2D* multiplyByTransposition(Tensor2D* tensor, Tensor2D* output);
    Tensor2D* transposeAndMultiply(Tensor2D* tensor, Tensor2D* output);
    Tensor1D* avg(Tensor1D* output);

	
};
#endif