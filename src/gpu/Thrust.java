package gpu;

import static gpu.ThrustNative.*;

import com.googlecode.javacpp.IntPointer;
import com.googlecode.javacpp.Loader;


/**
 * Wrapper around ThrustNative native methods.
 * Transformation methods are defined in pairs
 * exp(x): in-place transformation
 * exp(x, out): x immutable and store the result in out. Return the output parameter
 * a * x + b, default a = 1 and b = 0
 */
public class Thrust
{
	static { Loader.load(); }
	/**
	 * exp(a * x + b)
	 */
	public static void exp(FloatMat x, float a, float b)
	{
		gpu_exp_float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void exp(FloatMat x,  FloatMat out, float a, float b)
	{
		gpu_exp_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void exp(FloatMat x) {  exp(x, 1, 0); }
	public static void exp(FloatMat x, FloatMat out) {  exp(x, out, 1, 0); }
	
	/**
	 * ln(a * x + b)
	 */
	public static void log(FloatMat x, float a, float b)
	{
		gpu_log_float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void log(FloatMat x,  FloatMat out, float a, float b)
	{
		gpu_log_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void log(FloatMat x) {  log(x, 1, 0); }
	public static void log(FloatMat x, FloatMat out) {  log(x, out, 1, 0); }
	
	/**
	 * cos(a * x + b)
	 */
	public static void cos(FloatMat x, float a, float b)
	{
		gpu_cos_float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void cos(FloatMat x,  FloatMat out, float a, float b)
	{
		gpu_cos_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void cos(FloatMat x) {  cos(x, 1, 0); }
	public static void cos(FloatMat x, FloatMat out) {  cos(x, out, 1, 0); }
	
	/**
	 * sin(a * x + b)
	 */
	public static void sin(FloatMat x, float a, float b)
	{
		gpu_sin_float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void sin(FloatMat x,  FloatMat out, float a, float b)
	{
		gpu_sin_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void sin(FloatMat x) {  sin(x, 1, 0); }
	public static void sin(FloatMat x, FloatMat out) {  sin(x, out, 1, 0); }
	
	/**
	 * sqrt(a * x + b)
	 */
	public static void sqrt(FloatMat x, float a, float b)
	{
		gpu_sqrt_float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void sqrt(FloatMat x,  FloatMat out, float a, float b)
	{
		gpu_sqrt_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void sqrt(FloatMat x) {  sqrt(x, 1, 0); }
	public static void sqrt(FloatMat x, FloatMat out) {  sqrt(x, out, 1, 0); }
	
	/**
	 * (a * x + b) ^p
	 */
	public static void pow(FloatMat x, float p, float a, float b)
	{
		gpu_pow_float(x.getThrustPointer(), x.size(), p, a, b);
	}
	public static void pow(FloatMat x,  FloatMat out, float p, float a, float b)
	{
		gpu_pow_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), p, a, b);
	}
	public static void pow(FloatMat x, float p) {  pow(x, p, 1, 0); }
	public static void pow(FloatMat x, FloatMat out, float p) {  pow(x, out, p, 1, 0); }
	
	/**
	 * (a * x + b)
	 */
	public static void linear(FloatMat x, float a, float b)
	{
		gpu__float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void linear(FloatMat x,  FloatMat out, float a, float b)
	{
		gpu__float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	
	/**
	 * |a * x + b|
	 */
	public static void abs(FloatMat x, float a, float b)
	{
		gpu_fabs_float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void abs(FloatMat x,  FloatMat out, float a, float b)
	{
		gpu_fabs_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void abs(FloatMat x) {  abs(x, 1, 0); }
	public static void abs(FloatMat x, FloatMat out) {  abs(x, out, 1, 0); }
	
	public static float sum(FloatMat x)
	{
		return gpu_sum_float(x.getThrustPointer(), x.size());
	}
	
	public static float product(FloatMat x)
	{
		return gpu_product_float(x.getThrustPointer(), x.size());
	}
	
	public static float max(FloatMat x)
	{
		return gpu_max_float(x.getThrustPointer(), x.size());
	}
	
	public static float min(FloatMat x)
	{
		return gpu_min_float(x.getThrustPointer(), x.size());
	}
	
	
	/**
	 * Sort. dir = 1 for ascending, -1 for descending
	 * Default: ascending
	 */
	public static void sort(FloatMat x, int dir)
	{
		gpu_sort_float(x.getThrustPointer(), x.size(), dir);
	}
	/**
	 * Ascending sort
	 */
	public static void sort(FloatMat x) {	 sort(x, 1);	}
	
	/**
	 * Copy x to out
	 */
	public static void copy(FloatMat x, FloatMat out)
	{
		gpu_copy_float(x.getThrustPointer(), x.size(), out.getThrustPointer());
	}
	
	/**
	 * Swap x with y
	 */
	public static void swap(FloatMat x, FloatMat y)
	{
		gpu_swap_float(x.getThrustPointer(), x.size(), y.getThrustPointer());
	}
	
	/**
	 * Fill x with val
	 */
	public static void fill(FloatMat x, float val)
	{
		gpu_fill_float(x.getThrustPointer(), x.size(), val);
	}
	
	 // ******************** Babel specific methods ****************** /
    /**
     * I[y == j] - softmax(alpha_vec)
     */
	public static void babel_id_minus_softmax_float(FloatMat x, int id)
	{
		ThrustNative.babel_id_minus_softmax_float(x.getThrustPointer(), x.size(), id);
	}
	// version 2: more calculation, might be more numerically stable
	public static void babel_id_minus_softmax_float_2(FloatMat x, int id)
	{
		ThrustNative.babel_id_minus_softmax_float_2(x.getThrustPointer(), x.size(), id);
	}
	
	/**
	 * Minibatch: I[y == j] - softmax(alpha_vec)
	 */
	public static void babel_batch_id_minus_softmax_float(FloatMat x, IntPointer labels)
	{
		ThrustNative.babel_batch_id_minus_softmax_float(x.getThrustPointer(), x.row, x.col, labels);
	}
	// helper
	public static IntPointer copy_host_to_device(int[] labels)
	{
		return ThrustNative.copy_host_to_device(new IntPointer(labels), labels.length);
	}
	public static void gpu_free(IntPointer device)
	{
		ThrustNative.gpu_free(device);
	}
	
	
	//**************************************************/
	//******************* DOUBLE *******************/
	//**************************************************/
	/**
	 * exp(a * x + b)
	 */
	public static void exp(DoubleMat x, double a, double b)
	{
		gpu_exp_double(x.getThrustPointer(), x.size(), a, b);
	}
	public static void exp(DoubleMat x,  DoubleMat out, double a, double b)
	{
		gpu_exp_double(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void exp(DoubleMat x) {  exp(x, 1, 0); }
	public static void exp(DoubleMat x, DoubleMat out) {  exp(x, out, 1, 0); }
	
	/**
	 * ln(a * x + b)
	 */
	public static void log(DoubleMat x, double a, double b)
	{
		gpu_log_double(x.getThrustPointer(), x.size(), a, b);
	}
	public static void log(DoubleMat x,  DoubleMat out, double a, double b)
	{
		gpu_log_double(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void log(DoubleMat x) {  log(x, 1, 0); }
	public static void log(DoubleMat x, DoubleMat out) {  log(x, out, 1, 0); }
	
	/**
	 * cos(a * x + b)
	 */
	public static void cos(DoubleMat x, double a, double b)
	{
		gpu_cos_double(x.getThrustPointer(), x.size(), a, b);
	}
	public static void cos(DoubleMat x,  DoubleMat out, double a, double b)
	{
		gpu_cos_double(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void cos(DoubleMat x) {  cos(x, 1, 0); }
	public static void cos(DoubleMat x, DoubleMat out) {  cos(x, out, 1, 0); }
	
	/**
	 * sin(a * x + b)
	 */
	public static void sin(DoubleMat x, double a, double b)
	{
		gpu_sin_double(x.getThrustPointer(), x.size(), a, b);
	}
	public static void sin(DoubleMat x,  DoubleMat out, double a, double b)
	{
		gpu_sin_double(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void sin(DoubleMat x) {  sin(x, 1, 0); }
	public static void sin(DoubleMat x, DoubleMat out) {  sin(x, out, 1, 0); }
	
	/**
	 * sqrt(a * x + b)
	 */
	public static void sqrt(DoubleMat x, double a, double b)
	{
		gpu_sqrt_double(x.getThrustPointer(), x.size(), a, b);
	}
	public static void sqrt(DoubleMat x,  DoubleMat out, double a, double b)
	{
		gpu_sqrt_double(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void sqrt(DoubleMat x) {  sqrt(x, 1, 0); }
	public static void sqrt(DoubleMat x, DoubleMat out) {  sqrt(x, out, 1, 0); }
	
	/**
	 * (a * x + b) ^p
	 */
	public static void pow(DoubleMat x, double p, double a, double b)
	{
		gpu_pow_double(x.getThrustPointer(), x.size(), p, a, b);
	}
	public static void pow(DoubleMat x,  DoubleMat out, double p, double a, double b)
	{
		gpu_pow_double(x.getThrustPointer(), x.size(), out.getThrustPointer(), p, a, b);
	}
	public static void pow(DoubleMat x, double p) {  pow(x, p, 1, 0); }
	public static void pow(DoubleMat x, DoubleMat out, double p) {  pow(x, out, p, 1, 0); }
	
	/**
	 * (a * x + b)
	 */
	public static void linear(DoubleMat x, double a, double b)
	{
		gpu__double(x.getThrustPointer(), x.size(), a, b);
	}
	public static void linear(DoubleMat x,  DoubleMat out, double a, double b)
	{
		gpu__double(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	
	/**
	 * |a * x + b|
	 */
	public static void abs(DoubleMat x, double a, double b)
	{
		gpu_fabs_double(x.getThrustPointer(), x.size(), a, b);
	}
	public static void abs(DoubleMat x,  DoubleMat out, double a, double b)
	{
		gpu_fabs_double(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void abs(DoubleMat x) {  abs(x, 1, 0); }
	public static void abs(DoubleMat x, DoubleMat out) {  abs(x, out, 1, 0); }
	
	public static double sum(DoubleMat x)
	{
		return gpu_sum_double(x.getThrustPointer(), x.size());
	}
	
	public static double product(DoubleMat x)
	{
		return gpu_product_double(x.getThrustPointer(), x.size());
	}
	
	public static double max(DoubleMat x)
	{
		return gpu_max_double(x.getThrustPointer(), x.size());
	}
	
	public static double min(DoubleMat x)
	{
		return gpu_min_double(x.getThrustPointer(), x.size());
	}
	
	/**
	 * Sort. dir = 1 for ascending, -1 for descending
	 * Default: ascending
	 */
	public static void sort(DoubleMat x, int dir)
	{
		gpu_sort_double(x.getThrustPointer(), x.size(), dir);
	}
	/**
	 * Ascending sort
	 */
	public static void sort(DoubleMat x) {	 sort(x, 1);	}
	
	/**
	 * Copy x to out
	 */
	public static void copy(DoubleMat x, DoubleMat out)
	{
		gpu_copy_double(x.getThrustPointer(), x.size(), out.getThrustPointer());
	}
	
	/**
	 * Swap x with y
	 */
	public static void swap(DoubleMat x, DoubleMat y)
	{
		gpu_swap_double(x.getThrustPointer(), x.size(), y.getThrustPointer());
	}
	
	/**
	 * Fill x with val
	 */
	public static void fill(DoubleMat x, double val)
	{
		gpu_fill_double(x.getThrustPointer(), x.size(), val);
	}
	
	 // ******************** Babel specific methods ****************** /
    /**
     * I[y == j] - softmax(alpha_vec)
     */
	public static void babel_id_minus_softmax_double(DoubleMat x, int id)
	{
		ThrustNative.babel_id_minus_softmax_double(x.getThrustPointer(), x.size(), id);
	}
	
}
