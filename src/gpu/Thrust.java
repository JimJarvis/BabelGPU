package gpu;

import static gpu.ThrustNative.*;
import gpu.ThrustStruct.FloatDevicePointer;

import com.googlecode.javacpp.annotation.ByVal;

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
	public static FloatMat exp(FloatMat x, float a, float b)
	{
		ThrustNative.gpu_exp_float(x.getThrustPointer(), x.size(), a, b);
		return x;
	}
	public static FloatMat exp(FloatMat x,  FloatMat out, float a, float b)
	{
		ThrustNative.gpu_exp_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
		return out;
	}
	public static FloatMat exp(FloatMat x) { return exp(x, 1, 0); }
	public static FloatMat exp(FloatMat x, FloatMat out) { return exp(x, out, 1, 0); }
	
	/**
	 * ln(a * x + b)
	 */
	public static FloatMat log(FloatMat x, float a, float b)
	{
		ThrustNative.gpu_log_float(x.getThrustPointer(), x.size(), a, b);
		return x;
	}
	public static FloatMat log(FloatMat x,  FloatMat out, float a, float b)
	{
		ThrustNative.gpu_log_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
		return out;
	}
	public static FloatMat log(FloatMat x) { return log(x, 1, 0); }
	public static FloatMat log(FloatMat x, FloatMat out) { return log(x, out, 1, 0); }
	
	/**
	 * cos(a * x + b)
	 */
	public static FloatMat cos(FloatMat x, float a, float b)
	{
		ThrustNative.gpu_cos_float(x.getThrustPointer(), x.size(), a, b);
		return x;
	}
	public static FloatMat cos(FloatMat x,  FloatMat out, float a, float b)
	{
		ThrustNative.gpu_cos_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
		return out;
	}
	public static FloatMat cos(FloatMat x) { return cos(x, 1, 0); }
	public static FloatMat cos(FloatMat x, FloatMat out) { return cos(x, out, 1, 0); }
	
	/**
	 * sin(a * x + b)
	 */
	public static FloatMat sin(FloatMat x, float a, float b)
	{
		ThrustNative.gpu_sin_float(x.getThrustPointer(), x.size(), a, b);
		return x;
	}
	public static FloatMat sin(FloatMat x,  FloatMat out, float a, float b)
	{
		ThrustNative.gpu_sin_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
		return out;
	}
	public static FloatMat sin(FloatMat x) { return sin(x, 1, 0); }
	public static FloatMat sin(FloatMat x, FloatMat out) { return sin(x, out, 1, 0); }
	
	/**
	 * sqrt(a * x + b)
	 */
	public static FloatMat sqrt(FloatMat x, float a, float b)
	{
		ThrustNative.gpu_sqrt_float(x.getThrustPointer(), x.size(), a, b);
		return x;
	}
	public static FloatMat sqrt(FloatMat x,  FloatMat out, float a, float b)
	{
		ThrustNative.gpu_sqrt_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
		return out;
	}
	public static FloatMat sqrt(FloatMat x) { return sqrt(x, 1, 0); }
	public static FloatMat sqrt(FloatMat x, FloatMat out) { return sqrt(x, out, 1, 0); }
	
	/**
	 * (a * x + b) ^p
	 */
	public static FloatMat pow(FloatMat x, float p, float a, float b)
	{
		ThrustNative.gpu_pow_float(x.getThrustPointer(), x.size(), p, a, b);
		return x;
	}
	public static FloatMat pow(FloatMat x,  FloatMat out, float p, float a, float b)
	{
		ThrustNative.gpu_pow_float(x.getThrustPointer(), x.size(), out.getThrustPointer(), p, a, b);
		return out;
	}
	public static FloatMat pow(FloatMat x, float p) { return pow(x, p, 1, 0); }
	public static FloatMat pow(FloatMat x, FloatMat out, float p) { return pow(x, out, p, 1, 0); }
	
	/**
	 * (a * x + b)
	 */
	public static FloatMat linear(FloatMat x, float a, float b)
	{
		ThrustNative.gpu__float(x.getThrustPointer(), x.size(), a, b);
		return x;
	}
	public static FloatMat linear(FloatMat x,  FloatMat out, float a, float b)
	{
		ThrustNative.gpu__float(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
		return out;
	}
	public static FloatMat linear(FloatMat x) { return linear(x, 1, 0); }
	public static FloatMat linear(FloatMat x, FloatMat out) { return linear(x, out, 1, 0); }
	
	public static float sum(FloatMat x)
	{
		return ThrustNative.gpu_sum_float(x.getThrustPointer(), x.size());
	}
	
	public static float product(FloatMat x)
	{
		return ThrustNative.gpu_product_float(x.getThrustPointer(), x.size());
	}
	
	public static float max(FloatMat x)
	{
		return ThrustNative.gpu_max_float(x.getThrustPointer(), x.size());
	}
	
	public static float min(FloatMat x)
	{
		return ThrustNative.gpu_min_float(x.getThrustPointer(), x.size());
	}
	
	/**
	 * Sort. dir = 1 for ascending, -1 for descending
	 * Default: ascending
	 */
	public static FloatMat sort(FloatMat x, int dir)
	{
		ThrustNative.gpu_sort_float(x.getThrustPointer(), x.size(), dir);
		return x;
	}
	/**
	 * Ascending sort
	 */
	public static FloatMat sort(FloatMat x) {	return sort(x, 1);	}
	
	/**
	 * Copy x to out
	 */
	public static void copy(FloatMat x, FloatMat out)
	{
		ThrustNative.gpu_copy_float(x.getThrustPointer(), x.size(), out.getThrustPointer());
	}
	
	/**
	 * Swap x with y
	 */
	public static void swap(FloatMat x, FloatMat y)
	{
		ThrustNative.gpu_swap_float(x.getThrustPointer(), x.size(), y.getThrustPointer());
	}
	
	/**
	 * Fill x with val
	 */
	public static FloatMat fill(FloatMat x, float val)
	{
		ThrustNative.gpu_fill_float(x.getThrustPointer(), x.size(), val);
		return x;
	}
	
	 // ******************** Babel specific methods ****************** /
    /**
     * I[y == j] - softmax(alpha_vec)
     */
	public static void babel_id_minus_softmax(FloatMat x, int id)
	{
		ThrustNative.babel_id_minus_softmax(x.getThrustPointer(), x.size(), id);
	}
	
}
