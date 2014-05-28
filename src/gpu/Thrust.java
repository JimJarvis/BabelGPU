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
	public static void exp(FloatMat x, float a, float b) throws GpuException
	{
		gpu_exp_float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void exp(FloatMat x,  FloatMat out, float a, float b) throws GpuException
	{
		gpu_exp_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void exp(FloatMat x) throws GpuException
	{  
		exp(x, 1, 0); 
	}
	public static void exp(FloatMat x, FloatMat out) throws GpuException 
	{  
		exp(x, out, 1, 0); 
	}
	
	/**
	 * ln(a * x + b)
	 */
	public static void log(FloatMat x, float a, float b) throws GpuException
	{
		gpu_log_float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void log(FloatMat x,  FloatMat out, float a, float b) throws GpuException
	{
		gpu_log_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void log(FloatMat x)  throws GpuException
	{  
		log(x, 1, 0); 
	}
	public static void log(FloatMat x, FloatMat out)  throws GpuException
	{  
		log(x, out, 1, 0); 
	}
	
	/**
	 * cos(a * x + b)
	 * @throws GpuException 
	 */
	public static void cos(FloatMat x, int size, float a, float b) throws GpuException
	{
		gpu_cos_float(x.getThrustPointer(), size, a, b);
	}
	public static void cos(FloatMat x, float a, float b) throws GpuException
	{
		gpu_cos_float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void cos(FloatMat x,  FloatMat out, int size, float a, float b) throws GpuException
	{
		gpu_cos_float(x.getThrustPointer(), size, out.getThrustPointer(), a, b);
	}
	public static void cos(FloatMat x,  FloatMat out, float a, float b) throws GpuException
	{
		gpu_cos_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void cos(FloatMat x) throws GpuException 
	{
		cos(x, 1, 0); 
	}
	public static void cos(FloatMat x, FloatMat out) throws GpuException 
	{
		cos(x, out, 1, 0);
	}
	
	/**
	 * sin(a * x + b)
	 * @throws GpuException 
	 */
	public static void sin(FloatMat x, float a, float b) throws GpuException
	{
		gpu_sin_float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void sin(FloatMat x,  FloatMat out, float a, float b) throws GpuException
	{
		gpu_sin_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void sin(FloatMat x) throws GpuException 
	{
		sin(x, 1, 0); 
	}
	public static void sin(FloatMat x, FloatMat out)  throws GpuException
	{
		sin(x, out, 1, 0);
	}
	
	/**
	 * sqrt(a * x + b)
	 */
	public static void sqrt(FloatMat x, float a, float b)  throws GpuException
	{
		gpu_sqrt_float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void sqrt(FloatMat x,  FloatMat out, float a, float b) throws GpuException
	{
		gpu_sqrt_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void sqrt(FloatMat x) throws GpuException
	{
		sqrt(x, 1, 0); 
	}
	public static void sqrt(FloatMat x, FloatMat out) throws GpuException
	{
		sqrt(x, out, 1, 0); 
	}
	
	/**
	 * (a * x + b) ^p
	 */
	public static void pow(FloatMat x, float p, float a, float b) throws GpuException
	{
		gpu_pow_float(x.getThrustPointer(), x.size(), p, a, b);
	}
	public static void pow(FloatMat x,  FloatMat out, float p, float a, float b) throws GpuException
	{
		gpu_pow_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), p, a, b);
	}
	public static void pow(FloatMat x, float p) throws GpuException 
	{
		pow(x, p, 1, 0); 
	}
	public static void pow(FloatMat x, FloatMat out, float p) throws GpuException 
	{
		pow(x, out, p, 1, 0); 
	}
	
	/**
	 * (a * x + b)
	 */
	public static void linear(FloatMat x, int size, float a, float b) throws GpuException
	{
		gpu__float(x.getThrustPointer(), size, a, b);
	}
	public static void linear(FloatMat x, float a, float b) throws GpuException
	{
		gpu__float(x.getThrustPointer(), x.size(), a, b);
	}
	public static void linear(FloatMat x, FloatMat out, float a, float b) throws GpuException
	{
		gpu__float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	public static void linear(FloatMat x, FloatMat out, int size, float a, float b) throws GpuException
	{
		gpu__float(x.getThrustPointer(), size, out.getThrustPointer(), a, b);
	}
	
	public static float sum(FloatMat x) throws GpuException
	{
		return gpu_sum_float(x.getThrustPointer(), x.size());
	}
	
	public static float product(FloatMat x) throws GpuException
	{
		return gpu_product_float(x.getThrustPointer(), x.size());
	}
	
	public static float max(FloatMat x) throws GpuException
	{
		return gpu_max_float(x.getThrustPointer(), x.size());
	}
	
	public static float min(FloatMat x) throws GpuException
	{
		return gpu_min_float(x.getThrustPointer(), x.size());
	}
	
	/**
	 * Sort. dir = 1 for ascending, -1 for descending
	 * Default: ascending
	 */
	public static void sort(FloatMat x, int dir) throws GpuException
	{
		gpu_sort_float(x.getThrustPointer(), x.size(), dir);
	}
	/**
	 * Ascending sort
	 */
	public static void sort(FloatMat x) throws GpuException 
	{
		sort(x, 1);	
	}
	
	/**
	 * Copy x to out
	 */
	public static void copy(FloatMat x, FloatMat out) throws GpuException
	{
		gpu_copy_float(x.getThrustPointer(), x.size(), out.getThrustPointer());
	}
	
	/**
	 * Swap x with y
	 */
	public static void swap(FloatMat x, FloatMat y) throws GpuException
	{
		gpu_swap_float(x.getThrustPointer(), x.size(), y.getThrustPointer());
	}
	
	/**
	 * Fill x with val
	 */
	public static void fill(FloatMat x, float val) throws GpuException
	{
		gpu_fill_float(x.getThrustPointer(), x.size(), val);
	}
	
	 // ******************** Babel specific methods ****************** /
    /**
     * I[y == j] - softmax(alpha_vec)
     */
	public static void babel_id_minus_softmax(FloatMat x, int id) throws GpuException
	{
		ThrustNative.babel_id_minus_softmax(x.getThrustPointer(), x.size(), id);
	}
	// version 2: more calculation, might be more numerically stable
	public static void babel_id_minus_softmax_2(FloatMat x, int id) throws GpuException
	{
		ThrustNative.babel_id_minus_softmax_2(x.getThrustPointer(), x.size(), id);
	}
	
}
