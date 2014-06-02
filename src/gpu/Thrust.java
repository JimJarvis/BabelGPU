package gpu;

import static gpu.ThrustNative.*;

import com.googlecode.javacpp.IntPointer;
import com.googlecode.javacpp.Loader;
import com.googlecode.javacpp.annotation.*;


/**
 * Wrapper around ThrustNative native methods.
 * Transformation methods are defined in pairs
 * exp(x): in-place transformation
 * exp(x, out): x immutable and store the result in out. Return the output parameter
 * a * x + b, default a = 1 and b = 0
 */
@Platform(include={"\"my_gpu.h\"", "\"babel_gpu.h\""})
@Namespace("MyGpu")
public class Thrust
{
	static { Loader.load(); }

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
	
	 /**
     *  Set a specified row of a column-major matrix to be the same value
     *  @param rowIdx like python, wrapped around: if negative, rowIdx = rowDim + rowIdx
     */
    public static void fill_row(FloatMat x, int rowIdx, float val) throws GpuException
    {
    	ThrustNative.gpu_fill_row_float(x.getThrustPointer(), x.numRows, x.numCols, rowIdx, val);
    }
    /**
     *  Set a specified col of a  column-major matrix to be the same value
     *  @param colIdx like python, wrapped around: if negative, colIdx = colDim + colIdx
     */
    public static void fill_col(FloatMat x, int colIdx, float val) throws GpuException
    {
    	ThrustNative.gpu_fill_col_float(x.getThrustPointer(), x.numRows, x.numCols, colIdx, val);
    }
	
	 // ******************** Babel specific methods ****************** /
    /**
     * I[y == j] - softmax(alpha_vec)
     */
	public static void babel_id_minus_softmax_float(FloatMat x, int id) throws GpuException
	{
		ThrustNative.babel_id_minus_softmax(x.getThrustPointer(), x.size(), id);
	}
	
	/**
	 * Minibatch: I[y == j] - softmax(alpha_vec)
	 * @throws GpuException 
	 */
	public static void babel_batch_id_minus_softmax_float(FloatMat x, IntPointer labels) throws GpuException
	{
		ThrustNative.babel_batch_id_minus_softmax(x.getThrustPointer(), x.numRows, x.numCols, labels);
	}
	// helper
	public static IntPointer copy_host_to_device(int[] labels)
	{
		return ThrustNative.copy_host_to_device(new IntPointer(labels), labels.length);
	}
	
	   // Set the last row of a matrix to 1
    public static void set_last_row_one(FloatMat x) throws GpuException
    {
    	fill_row(x, -1, 1);
    }
    
    /**
     * Minibatch: softmax(alpha_vec)
     * @throws GpuException 
     */
    public static void babel_batch_softmax(FloatMat x) throws GpuException
    {
    	ThrustNative.babel_batch_softmax(x.getThrustPointer(), x.numRows, x.numCols);
    }

    /**
     * Minibatch: softmax(alpha_vec)
     * @param out writes to 'out' with probability only at the correct label of a column
	 * @param labels must be already on GPU. Call copy_host_to_device().
     * @throws GpuException 
     */
    public static void babel_batch_softmax(FloatMat x, FloatMat out, IntPointer labels) throws GpuException
    {
    	ThrustNative.babel_batch_softmax(
    			x.getThrustPointer(), x.numRows, x.numCols, out.getThrustPointer(), labels);
    }
    
    /**
     * Minibatch: get the labels where the maximum probability occurs
     * @param reusedDevicePtr use malloc_device() once to malloc on GPU
     * @param outLabels collects the maximum labels, 
     * 				writing from 'offset', write number of labels == label of columns
     * @throws GpuException 
     */
    public static void babel_best_label(
    		FloatMat x, IntPointer reusedDevicePtr, int[] outLabels, int offset) throws GpuException
	{
    	ThrustNative.babel_best_label(x.getThrustPointer(), x.numRows, x.numCols, reusedDevicePtr);
    	ThrustNative.copy_device_to_host(reusedDevicePtr, outLabels, offset, x.numCols);
	}

   
    // A few duplicates from ThrustNative.java
	// Force Thrust.java to generate code by JavaCpp
    public static native @ByPtr IntPointer copy_device_to_host(@ByPtr IntPointer device, int size);
    /**
     * Copy from device pointer directly to a host array, starting from 'offset'
     */
    public static native void copy_device_to_host(@ByPtr IntPointer device, @ByPtr int[] host, int offset, int size);
    
    public static native @ByPtr IntPointer malloc_device_int(int size, boolean memsetTo0);
    /**
     * @param memsetTo0 default false
     */
    public static IntPointer malloc_device_int(int size) {	return malloc_device_int(size, false); }
    
    public static native void free_device(@ByPtr IntPointer device);
    public static native void free_host(@ByPtr IntPointer host);
	public static native @ByPtr IntPointer offset(@ByPtr IntPointer begin, int offset);
}
