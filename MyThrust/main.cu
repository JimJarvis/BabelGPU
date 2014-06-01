#include "my_gpu.h"
#include "babel_gpu.h"
#include <iostream>
#define PI 3.14159265358979
#define range &D[0],D.size()
#define pr(stuff) std::cout << stuff << std::endl
using namespace MyGpu;

template<typename T>
void printD(T D)
{
	for (int i = 0; i < D.size(); i++)
		pr( "D[" << i << "] = " << D[i]);
}

// column major printing
template<typename T>
void printD(T D, int row)
{
	int col = D.size() / row;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			std::cout << D[i + j * row] << '\t';
		std::cout << std::endl;
	}
}


device_vector<float> getDf(float A[], int len)
{
	host_vector<float> D(A, A+len);
	return D;
}

device_vector<double> getDd(double A[], int len)
{
	host_vector<double> D(A, A + len);
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
	printD(D);

	printf("exp\n");
	printf("x\n");
	D = A; gpu_exp_float(range, 1, 0); printD(D);
	printf("0.5 * x\n");
	D = A; gpu_exp_float(range, 0.5, 0); printD(D);
	printf("x + 3\n");
	D = A; gpu_exp_float(range, 1, 3); printD(D);
	printf("0.5 * x + 3\n");
	D = A; gpu_exp_float(range, 0.5, 3); printD(D);

	printf("pow and sqrt\n");
	printf("x\n");
	float x[4] = { 25, 100, 16, 1024 };
	device_vector<float> F = getDf(x, 4);
	D = F; gpu_pow_float(range, 0.5, 1, 0); printD(D);
	D = F; gpu_sqrt_float(range, 1, 0); printD(D); // sqrt() should be the same as pow(x, 0.5)
	printf("0.7 * x\n");
	D = F; gpu_pow_float(range, 0.5, 0.7, 0); printD(D);
	D = F; gpu_sqrt_float(range, 0.7, 0); printD(D);
	printf("x + 4\n");
	D = F; gpu_pow_float(range, 0.5, 1, 4); printD(D);
	D = F; gpu_sqrt_float(range, 1, 4); printD(D);
	printf("0.7 * x + 4\n");
	D = F; gpu_pow_float(range, 0.5, 0.7, 4); printD(D);
	D = F; gpu_sqrt_float(range, 0.7, 4); printD(D);
}

void test_babel()
{
	float x[7] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6 };
	device_vector<float> D = getDf(x, 7);
	pr(gpu_product_float(range));
	pr(gpu_min_float(range));

	babel_id_minus_softmax_float(range, 3);
	printD(D);
}

void test_sort_copy_swap()
{
	float x[7] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6 };
	device_vector<float> D = getDf(x, 7);
	gpu_sort_float(range);
	printD(D);

	D = getDf(x, 7);
	gpu_sort_float(range, -1);
	printD(D);

	device_vector<float> E(7, -666);
	printf("Copying E\n");
	gpu_copy_float(range, &E[0]);
	printD(E);
	gpu_fill_float(&E[0], E.size(), -666);
	gpu_copy_float(&D[2], 3, &E[4]);
	printD(E);

	printf("Swapping E\n");
	float y[4] = { 400, 300, 200, 100 };
	D = getDf(y, 4);
	gpu_swap_float(range, &E[0]);
	printD(D);
	printD(E);
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
	printD(E);

	printf("exp\n");
	printf("x\n");
	gpu_exp_float(range, &E[0], 1, 0); printD(E);
	printf("0.5 * x\n");
	gpu_exp_float(range, &E[0], 0.5, 0); printD(E);
	printf("x + 3\n");
	gpu_exp_float(range, &E[0], 1, 3); printD(E);
	printf("0.5 * x + 3\n");
	gpu_exp_float(range, &E[0], 0.5, 3); printD(E);
}


void test_exp_double()
{
	host_vector<double> A(4);
	A[0] = 3;
	A[1] = -2;
	A[2] = PI;
	A[3] = -PI / 3;

	device_vector<double> D = A;
	pr(gpu_max_double(range));

	gpu_cos_double(range);
	printD(D);

	printf("exp\n");
	printf("x\n");
	D = A; gpu_exp_double(range, 1, 0); printD(D);
	printf("0.5 * x\n");
	D = A; gpu_exp_double(range, 0.5, 0); printD(D);
	printf("x + 3\n");
	D = A; gpu_exp_double(range, 1, 3); printD(D);
	printf("0.5 * x + 3\n");
	D = A; gpu_exp_double(range, 0.5, 3); printD(D);

	printf("pow and sqrt\n");
	printf("x\n");
	double x[4] = { 25, 100, 16, 1024 };
	device_vector<double> F = getDd(x, 4);
	D = F; gpu_pow_double(range, 0.5, 1, 0); printD(D);
	D = F; gpu_sqrt_double(range, 1, 0); printD(D); // sqrt() should be the same as pow(x, 0.5)
	printf("0.7 * x\n");
	D = F; gpu_pow_double(range, 0.5, 0.7, 0); printD(D);
	D = F; gpu_sqrt_double(range, 0.7, 0); printD(D);
	printf("x + 4\n");
	D = F; gpu_pow_double(range, 0.5, 1, 4); printD(D);
	D = F; gpu_sqrt_double(range, 1, 4); printD(D);
	printf("0.7 * x + 4\n");
	D = F; gpu_pow_double(range, 0.5, 0.7, 4); printD(D);
	D = F; gpu_sqrt_double(range, 0.7, 4); printD(D);
}

void test_sort_copy_swap_double()
{
	double x[7] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6 };
	device_vector<double> D = getDd(x, 7);
	gpu_sort_double(range);
	printD(D);

	D = getDd(x, 7);
	gpu_sort_double(range, -1);
	printD(D);

	device_vector<double> E(7, -666);
	printf("Copying E\n");
	gpu_copy_double(range, &E[0]);
	printD(E);
	gpu_fill_double(&E[0], E.size(), -666);
	gpu_copy_double(&D[2], 3, &E[4]);
	printD(E);

	printf("Swapping E\n");
	double y[4] = { 400, 300, 200, 100 };
	D = getDd(y, 4);
	gpu_swap_double(range, &E[0]);
	printD(D);
	printD(E);
}

// babel mini-batch (I[] - softmax)
void test_batch()
{
	float x[12] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6, 2.5, 1.7, -0.2, -0.9, 0.4 };
	device_vector<float> D = getDf(x, 12);
	int labels[6] = {-100,-1000, 0, 3, 2, 1 };
	device_vector<int> L = host_vector<int>(labels, labels + 6);
	int *lp = thrust::raw_pointer_cast(&L[0]);
	lp = offset(lp, 2);

	//babel_batch_id_minus_softmax_float(&D[0], 4, 3, lp);
	babel_batch_id_minus_softmax_float(&D[0], 12, 1, lp);

	printD(D, 4);
}

// Set a row/col to a specific value
void test_set_row_col()
{
	float x[12] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6, 2.5, 1.7, -0.2, -0.9, 0.4 };
	device_vector<float> D = getDf(x, 12);
	gpu_fill_col_float(&D[0], 4, 3, -2, 600);
	gpu_fill_row_float(&D[0], 4, 3, -1, 100);
	printD(D, 4);
	D = getDf(x, 12);
	gpu_fill_row_float(&D[0], 4, 3, 3, -100);
	gpu_fill_col_float(&D[0], 4, 3, 2, -600);
	printD(D, 4);
}


void main()
{
	//test_set_row_col();
	test_batch();
	//test_babel();
	//test_exp_double();
	//test_sort_copy_swap_double();
	//test_exp();
	//test_exp_out_pointer();
	//test_sort_copy_swap();
}