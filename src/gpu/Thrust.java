package gpu;

import static gpu.ThrustNative.*;


/**
 * Wrapper around ThrustNative native methods.
 * Transformation methods are defined in pairs
 * exp(x): in-place transformation
 * exp(x, out): x immutable and store the result in out. Return the output parameter
 * a * x + b, default a = 1 and b = 0
 */
public class Thrust
{
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
	public static void babel_id_minus_softmax(FloatMat x, int id)
	{
		ThrustNative.babel_id_minus_softmax(x.getThrustPointer(), x.size(), id);
	}
	// version 2: more calculation, might be more numerically stable
	public static void babel_id_minus_softmax_2(FloatMat x, int id)
	{
		ThrustNative.babel_id_minus_softmax_2(x.getThrustPointer(), x.size(), id);
	}
	
}
