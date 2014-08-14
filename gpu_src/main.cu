#include "my_gpu.h"
#include "my_kernel.h"
#include <iostream>
#define range &D[0],D.size()
#define pr(stuff) std::cout << stuff << std::endl
using namespace MyGpu;

template<typename T>
void printD(T D)
{
	for (int i = 0; i < D.size(); i++)
		pr( "D[" << i << "] = " << D[i]);
}
template<typename T>
void printH(T D, int size)
{
	for (int i = 0; i < size; i++)
		pr("H[" << i << "] = " << D[i]);
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

// gpu_exp<float>() in place transformation
void test_exp()
{
	host_vector<float> A(4);
	A[0] = 3;
	A[1] = -2;
	A[2] = PI;
	A[3] = -PI/3;

	device_vector<float> D = A;
	pr( gpu_max<float>(range) );

	gpu_cos<float>(range);
	printD(D);

	printf("exp\n");
	printf("x\n");
	D = A; gpu_exp<float>(range, 1, 0); printD(D);
	printf("0.5 * x\n");
	D = A; gpu_exp<float>(range, 0.5, 0); printD(D);
	printf("x + 3\n");
	D = A; gpu_exp<float>(range, 1, 3); printD(D);
	printf("0.5 * x + 3\n");
	D = A; gpu_exp<float>(range, 0.5, 3); printD(D);

	printf("pow and sqrt\n");
	printf("x\n");
	float x[4] = { 25, 100, 16, 1024 };
	device_vector<float> F = getDf(x, 4);
	D = F; gpu_pow<float>(range, 0.5, 1, 0); printD(D);
	D = F; gpu_sqrt<float>(range, 1, 0); printD(D); // sqrt() should be the same as pow(x, 0.5)
	printf("0.7 * x\n");
	D = F; gpu_pow<float>(range, 0.5, 0.7, 0); printD(D);
	D = F; gpu_sqrt<float>(range, 0.7, 0); printD(D);
	printf("x + 4\n");
	D = F; gpu_pow<float>(range, 0.5, 1, 4); printD(D);
	D = F; gpu_sqrt<float>(range, 1, 4); printD(D);
	printf("0.7 * x + 4\n");
	D = F; gpu_pow<float>(range, 0.5, 0.7, 4); printD(D);
	D = F; gpu_sqrt<float>(range, 0.7, 4); printD(D);
}

void test_sort_copy_swap()
{
	float x[7] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6 };
	device_vector<float> D = getDf(x, 7);
	gpu_sort<float>(range);
	printD(D);

	D = getDf(x, 7);
	gpu_sort<float>(range, -1);
	printD(D);

	device_vector<float> E(7, -666);
	printf("Copying E\n");
	gpu_copy<float>(range, &E[0]);
	printD(E);
	gpu_fill<float>(&E[0], E.size(), -666);
	gpu_copy<float>(&D[2], 3, &E[4]);
	printD(E);

	printf("Swapping E\n");
	float y[4] = { 400, 300, 200, 100 };
	D = getDf(y, 4);
	gpu_swap<float>(range, &E[0]);
	printD(D);
	printD(E);
}

// gpu_exp<float> with output pointer
void test_exp_out_pointer()
{
	host_vector<float> A(4);
	A[0] = 3;
	A[1] = -2;
	A[2] = PI;
	A[3] = -PI / 3;

	device_vector<float> D = A;
	device_vector<float> E = A;

	gpu_cos<float>(range, &E[0]);
	printD(E);

	printf("exp\n");
	printf("x\n");
	gpu_exp<float>(range, &E[0], 1, 0); printD(E);
	printf("0.5 * x\n");
	gpu_exp<float>(range, &E[0], 0.5, 0); printD(E);
	printf("x + 3\n");
	gpu_exp<float>(range, &E[0], 1, 3); printD(E);
	printf("0.5 * x + 3\n");
	gpu_exp<float>(range, &E[0], 0.5, 3); printD(E);
}

void test_exp_double()
{
	host_vector<double> A(4);
	A[0] = 3;
	A[1] = -2;
	A[2] = PI;
	A[3] = -PI / 3;

	device_vector<double> D = A;
	pr(gpu_max<double>(range));

	gpu_cos<double>(range);
	printD(D);

	printf("exp\n");
	printf("x\n");
	D = A; gpu_exp<double>(range, 1, 0); printD(D);
	printf("0.5 * x\n");
	D = A; gpu_exp<double>(range, 0.5, 0); printD(D);
	printf("x + 3\n");
	D = A; gpu_exp<double>(range, 1, 3); printD(D);
	printf("0.5 * x + 3\n");
	D = A; gpu_exp<double>(range, 0.5, 3); printD(D);

	printf("pow and sqrt\n");
	printf("x\n");
	double x[4] = { 25, 100, 16, 1024 };
	device_vector<double> F = getDd(x, 4);
	D = F; gpu_pow<double>(range, 0.5, 1, 0); printD(D);
	D = F; gpu_sqrt<double>(range, 1, 0); printD(D); // sqrt() should be the same as pow(x, 0.5)
	printf("0.7 * x\n");
	D = F; gpu_pow<double>(range, 0.5, 0.7, 0); printD(D);
	D = F; gpu_sqrt<double>(range, 0.7, 0); printD(D);
	printf("x + 4\n");
	D = F; gpu_pow<double>(range, 0.5, 1, 4); printD(D);
	D = F; gpu_sqrt<double>(range, 1, 4); printD(D);
	printf("0.7 * x + 4\n");
	D = F; gpu_pow<double>(range, 0.5, 0.7, 4); printD(D);
	D = F; gpu_sqrt<double>(range, 0.7, 4); printD(D);
}

void test_sort_copy_swap_double()
{
	double x[7] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6 };
	device_vector<double> D = getDd(x, 7);
	gpu_sort<double>(range);
	printD(D);

	D = getDd(x, 7);
	gpu_sort<double>(range, -1);
	printD(D);

	device_vector<double> E(7, -666);
	printf("Copying E\n");
	gpu_copy<double>(range, &E[0]);
	printD(E);
	gpu_fill<double>(&E[0], E.size(), -666);
	gpu_copy<double>(&D[2], 3, &E[4]);
	printD(E);

	printf("Swapping E\n");
	double y[4] = { 400, 300, 200, 100 };
	D = getDd(y, 4);
	gpu_swap<double>(range, &E[0]);
	printD(D);
	printD(E);
}

// Set a row/col to a specific value
void test_set_row_col()
{
	float x[12] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6, 2.5, 1.7, -0.2, -0.9, 0.4 };
	device_vector<float> D = getDf(x, 12);
	gpu_fill_col<float>(&D[0], 4, 3, -2, 600);
	gpu_fill_row<float>(&D[0], 4, 3, -1, 100);
	printD(D, 4);
	D = getDf(x, 12);
	gpu_fill_row<float>(&D[0], 4, 3, 3, -100);
	gpu_fill_col<float>(&D[0], 4, 3, 2, -600);
	printD(D, 4);
}


// ****************** Babel specific tests ******************
void test_softmax_minus_id()
{
	float x[7] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6 };
	device_vector<float> D = getDf(x, 7);
	pr(gpu_product<float>(range));
	pr(gpu_min<float>(range));

	gpu_softmax_minus_id(range, &D[0], 3);
	printD(D);
}

// babel mini-batch (I[] - softmax)
void test_softmax_minus_id_batch()
{
	float x[12] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6, 2.5, 1.7, -0.2, -0.9, 0.4 };
	device_vector<float> D = getDf(x, 12);
	int labels[6] = {-100,-1000, 0, 3, 2, 1 };
	device_vector<int> L = host_vector<int>(labels, labels + 6);
	int *lp = thrust::raw_pointer_cast(&L[0]);
	lp = offset(lp, 2);

	gpu_batch_softmax_minus_id(&D[0], 4, 3, &D[0], lp, false);
	printD(D, 4);
}


// babel mini-batch (softmax probability distr)
void test_softmax_batch()
{
	float x[12] = { 4.2, 5.9, -2.1, -3.7, 3.3, 1.9, -0.6, 2.5, 1.7, -0.2, -0.9, 0.4 };
	device_vector<float> D = getDf(x, 12);

	printf("Unlabeled softmax\n");
	gpu_batch_softmax(&D[0], 4, 3, false);
	printD(D, 4);

	printf("Labeled softmax\n");
	int labels[4] = {0, 1, 2, 1 };
	device_vector<int> L = host_vector<int>(labels, labels + 4);
	int *lp = thrust::raw_pointer_cast(&L[0]);
	D = getDf(x, 12);
	device_vector<float> out(4);
	gpu_batch_softmax_at_label(&D[0], 3, 4, &out[0], lp, false);
	printD(out, 1);
	printf("Log likelihood sum: %f\n", 
		   gpu_log_sum(&out[0], out.size()));

	const int SIZE = 1946;
	float y[SIZE];
	for (int i = 0; i < SIZE; y[i] = float(rand()) / RAND_MAX, i++);
	D = getDf(y, SIZE);
	gpu_batch_softmax(&D[0], 2, SIZE / 2, false);

	device_vector<int> outLabels(4);
	float p[12] = {2, 5, 1, 3, -4, -2, 1, 0, 9, 3, -5, 6};
	D = getDf(p, 12);
	lp = thrust::raw_pointer_cast(&outLabels[0]);
	gpu_best_label(&D[0], 4, 3, lp, false);
	printD(outLabels, 1);

	int outLabelHost[10];
	copy_device_to_host(lp, outLabelHost, 5, 3);
	printH(outLabelHost, 10);
}

void test_sigmoid()
{
	host_vector<float> A(4);
	A[0] = 0;
	A[1] = -1;
	A[2] = 1;
	A[3] = 1e15;

	device_vector<float> D = A;
	device_vector<float> E = A;

	printf("sigmoid\n");
	printf("x\n");
	gpu_sigmoid<float>(range, &E[0], 1, 0); printD(E);
	gpu_sigmoid_deriv<float>(range, &E[0], 1, 0); printD(E);
	printf("0.5 * x\n");
	gpu_sigmoid<float>(range, &E[0], 0.5, 0); printD(E);
	printf("x + 3\n");
	gpu_sigmoid<float>(range, &E[0], 1, 3); printD(E);
	printf("0.5 * x + 3\n");
	gpu_sigmoid<float>(range, &E[0], 0.5, 3); printD(E);
}

void test_tranpose()
{
	float in[12] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
	device_vector<float> In = getDf(in, 12);
	float out[12];
	device_vector<float> Out = getDf(out, 12);
	printf("Original mat\n");
	printD(In, 4);
	gpu_transpose<float>(&In[0], 4, 3, &Out[0]);
	printf("Transposed mat 3 x 4\n");
	printD(Out, 3);
	gpu_transpose<float>(&In[0], 6, 2, &Out[0]);
	printf("Transposed mat 2 x 6\n");
	printD(Out, 2);
	gpu_transpose<float>(&In[0], 12, 1, &Out[0]);
	printf("Transposed mat 1 x 12\n");
	printD(Out, 1);
	gpu_transpose<float>(&In[0], 2, 6, &Out[0]);
	printf("Transposed mat 6 x 2\n");
	printD(Out, 6);
}

void test_rand_normal()
{
	//float in[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	//device_vector<float> In = getDf(in, 12);
	device_vector<float> In(12);
	gpu_fill_rand_normal<float>(&In[0], 9, 30.0, 2.5);
	printD(In, 12);
}

int main()
{
	//test_rand_normal();
	//test_tranpose();
	//test_sigmoid();
	//test_softmax_minus_id_batch();
	//test_softmax_batch();
	//test_set_row_col();
	//test_exp_double();
	//test_sort_copy_swap_double();
	//test_exp();
	//test_exp_out_pointer();
	//test_sort_copy_swap();
	device_vector<float> In(12);
	gpu_fill_rand_normal<float>(&In[0], 9, 30.0, 2.5);
	gpu_triangular_wave(&In[0], 10);
	printD(In, 12);

	return 0;
}