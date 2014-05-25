#include "my_gpu.h"
#include "babel_gpu.h"
#include <iostream>
#define PI 3.14159265358979
using namespace MyGpu;

void printDevice(device_vector<float> D)
{
	for (int i = 0; i < D.size(); i++)
		std::cout << "D[" << i << "] = " << D[i] << std::endl;
}

void main()
{
	host_vector<float> A(4);
	A[0] = 3;
	A[1] = -2;
	A[2] = PI;
	A[3] = -PI/3;

	device_vector<float> D = A;
	std::cout << gpu_max_float(&D[0], &D[4]) << std::endl;

	gpu_cos_float(&D[0], &D[4]);
	printDevice(D);

	printf("exp\n");
	printf("x\n");
	D = A; gpu_exp_float(&D[0], &D[4], 1, 0); printDevice(D);
	printf("0.5 * x\n");
	D = A; gpu_exp_float(&D[0], &D[4], 0.5, 0); printDevice(D);
	printf("x + 3\n");
	D = A; gpu_exp_float(&D[0], &D[4], 1, 3); printDevice(D);
	printf("0.5 * x + 3\n");
	D = A; gpu_exp_float(&D[0], &D[4], 0.5, 3); printDevice(D);

	A[0] = 25;
	A[1] = 100;
	A[2] = 16;
	A[3] = 1024;
	printf("pow and sqrt\n");
	printf("x\n");
	D = A; gpu_pow_float(&D[0], &D[4], 0.5, 1, 0); printDevice(D);
	D = A; gpu_sqrt_float(&D[0], &D[4], 1, 0); printDevice(D); // sqrt() should be the same as pow(x, 0.5)
	printf("0.7 * x\n");
	D = A; gpu_pow_float(&D[0], &D[4], 0.5, 0.7, 0); printDevice(D);
	D = A; gpu_sqrt_float(&D[0], &D[4], 0.7, 0); printDevice(D);
	printf("x + 4\n");
	D = A; gpu_pow_float(&D[0], &D[4], 0.5, 1, 4); printDevice(D);
	D = A; gpu_sqrt_float(&D[0], &D[4], 1, 4); printDevice(D);
	printf("0.7 * x + 4\n");
	D = A; gpu_pow_float(&D[0], &D[4], 0.5, 0.7, 4); printDevice(D);
	D = A; gpu_sqrt_float(&D[0], &D[4], 0.7, 4); printDevice(D);

	printf("Goody almighty %d", babel_id_minus_softmax(&D[0], &D[4], 3));
}