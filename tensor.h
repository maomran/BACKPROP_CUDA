#ifndef TENSOR_H
#define TENSOR_H

#include <cstdio>

class tensor
{
public:
	int idx;
	int idy;
	float *data;

	tensor(int idx, int idy);
	tensor(int idx, int idy, float *d_data);
	tensor(int idx, int idym float **h_data);
	~tensor();
	
};
#endif