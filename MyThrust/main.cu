#include "my_gpu.h"
#include "babel_gpu.h"
#include <iostream>
#define PI 3.14159265358979
#define range &D[0],D.size()
#define pr(stuff) std::cout << stuff << std::endl
using namespace MyGpu;

void printDevice(device_vector<float> D)
{
	for (int i = 0; i < D.size(); i++)
		pr( "D[" << i << "] = " << D[i]);
}

device_vector<float> getD(float A[], int len)
{
	host_vector<float> D(A, A+len);
	return D;
}

// gpu_exp_float() in place transformation
void test_exp()
{
	host_vector<float> A(4);
	A[0] = 3;
	A[1] = -2;
	A[2] = PI;
	A[3] = -PI/3;

	device_vector<float> D = A;
	pr( gpu_max_float(range) );

	gpu_cos_float(range);
	printDevice(D);

	printf("exp\n");
	printf("x\n");
	D = A; gpu_exp_float(range, 1, 0); printDevice(D);
	printf("0.5 * x\n");
	D = A; gpu_exp_float(range, 0.5, 0); printDevice(D);
	printf("x + 3\n");
	D = A; gpu_exp_float(range, 1, 3); printDevice(D);
	printf("0.5 * x + 3\n");
	D = A; gpu_exp_float(range, 0.5, 3); printDevice(D);

	printf("pow and sqrt\n");
	printf("x\n");
	float x[4] = { 25, 100, 16, 1024 };
	device_vector<float> F = getD(x, 4);
	D = F; gpu_pow_float(range, 0.5, 1, 0); printDevice(D);
	D = F; gpu_sqrt_float(range, 1, 0); printDevice(D); // sqrt() should be the same as pow(x, 0.5)
	printf("0.7 * x\n");
	D = F; gpu_pow_float(range, 0.5, 0.7, 0); printDevice(D);
	D = F; gpu_sqrt_float(range, 0.7, 0); printDevice(D);
	printf("x + 4\n");
	D = F; gpu_pow_float(range, 0.5, 1, 4); printDevice(D);
	D = F; gpu_sqrt_float(range, 1, 4); printDevice(D);
	printf("0.7 * x + 4\n");
	D = F; gpu_pow_float(range, 0.5, 0.7, 4); printDevice(D);
	D = F; gpu_sqrt_float(range, 0.7, 4); printDevice(D);
}

void test_babel()
{
	float x[7] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6 };
	device_vector<float> D = getD(x, 7);
	pr(gpu_product_float(range));
	pr(gpu_min_float(range));

	babel_id_minus_softmax(range, 3);
	printDevice(D);
}

void test_sort_copy_swap()
{
	float x[7] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6 };
	device_vector<float> D = getD(x, 7);
	gpu_sort_float(range);
	printDevice(D);

	D = getD(x, 7);
	gpu_sort_float(range, -1);
	printDevice(D);

	device_vector<float> E(7, -666);
	printf("Copying E\n");
	gpu_copy_float(range, &E[0]);
	printDevice(E);
	gpu_fill_float(&E[0], E.size(), -666);
	gpu_copy_float(&D[2], 3, &E[4]);
	printDevice(E);

	printf("Swapping E\n");
	float y[4] = { 400, 300, 200, 100 };
	D = getD(y, 4);
	gpu_swap_float(range, &E[0]);
	printDevice(D);
	printDevice(E);
}

// gpu_exp_float with output pointer
void test_exp_out_pointer()
{
	host_vector<float> A(4);
	A[0] = 3;
	A[1] = -2;
	A[2] = PI;
	A[3] = -PI / 3;

	device_vector<float> D = A;
	device_vector<float> E = A;

	gpu_cos_float(range, &E[0]);
	printDevice(E);

	printf("exp\n");
	printf("x\n");
	gpu_exp_float(range, &E[0], 1, 0); printDevice(E);
	printf("0.5 * x\n");
	gpu_exp_float(range, &E[0], 0.5, 0); printDevice(E);
	printf("x + 3\n");
	gpu_exp_float(range, &E[0], 1, 3); printDevice(E);
	printf("0.5 * x + 3\n");
	gpu_exp_float(range, &E[0], 0.5, 3); printDevice(E);
}

void main()
{
	test_babel();
	test_exp();
	test_exp_out_pointer();
	test_sort_copy_swap();
}