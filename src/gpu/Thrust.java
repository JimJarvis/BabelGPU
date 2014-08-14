package gpu;

import static gpu.Natives.*;
import gpu.NativeStruct.FloatDevicePointer;

import com.googlecode.javacpp.*;
import com.googlecode.javacpp.annotation.*;

/**
 * Wrapper around ThrustNative native methods.
 * Transformation methods are defined in pairs
 * exp(x): in-place transformation
 * exp(x, out): x immutable and store the result in out. Return the output parameter
 * a * x + b, default a = 1 and b = 0
 */
@Platform(include={"\"my_gpu.h\"", "\"my_kernel.h\""})
@Namespace("MyGpu")
public class Thrust
{
	static { Loader.load(); }
	/**
	 * m * exp(a * x + b)
	 */
	public static void exp(FloatMat x, float a, float b, float m)
	{
		gpu_exp(x.getThrustPointer(), x.size(), a, b, m);
	}
	/**
	 * m * exp(a * x + b)
	 */
	public static void exp(FloatMat x,  FloatMat out, float a, float b, float m)
	{
		gpu_exp(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b, m);
	}
	public static void exp(FloatMat x) {  exp(x, 1, 0, 1); }
	public static void exp(FloatMat x, FloatMat out) {  exp(x, out, 1, 0, 1); }
	
	/**
	 * m * ln(a * x + b)
	 */
	public static void log(FloatMat x, float a, float b, float m)
	{
		gpu_log(x.getThrustPointer(), x.size(), a, b, m);
	}
	/**
	 * m * ln(a * x + b)
	 */
	public static void log(FloatMat x,  FloatMat out, float a, float b, float m)
	{
		gpu_log(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b, m);
	}
	public static void log(FloatMat x) {  log(x, 1, 0, 1); }
	public static void log(FloatMat x, FloatMat out) {  log(x, out, 1, 0, 1); }
	
	/**
	 * m * cos(a * x + b)
	 */
	public static void cos(FloatMat x, float a, float b, float m)
	{
		gpu_cos(x.getThrustPointer(), x.size(), a, b, m);
	}
	/**
	 * m * cos(a * x + b)
	 */
	public static void cos(FloatMat x,  FloatMat out, float a, float b, float m)
	{
		gpu_cos(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b, m);
	}
	public static void cos(FloatMat x) {  cos(x, 1, 0, 1); }
	public static void cos(FloatMat x, FloatMat out) {  cos(x, out, 1, 0, 1); }
	
	/**
	 * m * sin(a * x + b)
	 */
	public static void sin(FloatMat x, float a, float b, float m)
	{
		gpu_sin(x.getThrustPointer(), x.size(), a, b, m);
	}
	/**
	 * m * sin(a * x + b)
	 */
	public static void sin(FloatMat x,  FloatMat out, float a, float b, float m)
	{
		gpu_sin(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b, m);
	}
	public static void sin(FloatMat x) {  sin(x, 1, 0, 1); }
	public static void sin(FloatMat x, FloatMat out) {  sin(x, out, 1, 0, 1); }
	
	/**
	 * m * sqrt(a * x + b)
	 */
	public static void sqrt(FloatMat x, float a, float b, float m)
	{
		gpu_sqrt(x.getThrustPointer(), x.size(), a, b, m);
	}
	/**
	 * m * sqrt(a * x + b)
	 */
	public static void sqrt(FloatMat x,  FloatMat out, float a, float b, float m)
	{
		gpu_sqrt(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b, m);
	}
	public static void sqrt(FloatMat x) {  sqrt(x, 1, 0, 1); }
	public static void sqrt(FloatMat x, FloatMat out) {  sqrt(x, out, 1, 0, 1); }
	
	/**
	 * m * (a * x + b) ^p
	 */
	public static void pow(FloatMat x, float p, float a, float b, float m)
	{
		gpu_pow(x.getThrustPointer(), x.size(), p, a, b, m);
	}
	/**
	 * m * (a * x + b) ^p
	 */
	public static void pow(FloatMat x,  FloatMat out, float p, float a, float b, float m)
	{
		gpu_pow(x.getThrustPointer(), x.size(), out.getThrustPointer(), p, a, b, m);
	}
	public static void pow(FloatMat x, float p) {  pow(x, p, 1, 0, 1); }
	public static void pow(FloatMat x, FloatMat out, float p) {  pow(x, out, p, 1, 0, 1); }
	
	/**
	 * m * (a * x + b)^2
	 */
	public static void square(FloatMat x, float a, float b, float m)
	{
		gpu_square(x.getThrustPointer(), x.size(), a, b, m);
	}
	/**
	 * m * (a * x + b)^2
	 */
	public static void square(FloatMat x,  FloatMat out, float a, float b, float m)
	{
		gpu_square(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b, m);
	}
	public static void square(FloatMat x) {  square(x, 1, 0, 1); }
	public static void square(FloatMat x, FloatMat out) {  square(x, out, 1, 0, 1); }

	/**
	 * m * (a * x + b)^3
	 */
	public static void cube(FloatMat x, float a, float b, float m)
	{
		gpu_cube(x.getThrustPointer(), x.size(), a, b, m);
	}
	/**
	 * m * (a * x + b)^3
	 */
	public static void cube(FloatMat x,  FloatMat out, float a, float b, float m)
	{
		gpu_cube(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b, m);
	}
	public static void cube(FloatMat x) {  cube(x, 1, 0, 1); }
	public static void cube(FloatMat x, FloatMat out) {  cube(x, out, 1, 0, 1); }
	
	/**
	 * m * 1 / (a * x + b)
	 */
	public static void reciprocal(FloatMat x, float a, float b, float m)
	{
		gpu_reciprocal(x.getThrustPointer(), x.size(), a, b, m);
	}
	/**
	 * m * 1 / (a * x + b)
	 */
	public static void reciprocal(FloatMat x,  FloatMat out, float a, float b, float m)
	{
		gpu_reciprocal(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b, m);
	}
	public static void reciprocal(FloatMat x) {  reciprocal(x, 1, 0, 1); }
	public static void reciprocal(FloatMat x, FloatMat out) {  reciprocal(x, out, 1, 0, 1); }
	
	/**
	 * (a * x + b)
	 */
	public static void linear(FloatMat x, float a, float b)
	{
		gpu_linear(x.getThrustPointer(), x.size(), a, b);
	}
	/**
	 * (a * x + b)
	 */
	public static void linear(FloatMat x,  FloatMat out, float a, float b)
	{
		gpu_linear(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b);
	}
	
	/**
	 * m * |a * x + b|
	 */
	public static void abs(FloatMat x, float a, float b, float m)
	{
		gpu_fabs(x.getThrustPointer(), x.size(), a, b, m);
	}
	/**
	 * m * |a * x + b|
	 */
	public static void abs(FloatMat x,  FloatMat out, float a, float b, float m)
	{
		gpu_fabs(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b, m);
	}
	public static void abs(FloatMat x) {  abs(x, 1, 0, 1); }
	public static void abs(FloatMat x, FloatMat out) {  abs(x, out, 1, 0, 1); }
	
	/**
	 * m * Sigmoid(a * x + b)
	 */
	public static void sigmoid(FloatMat x, float a, float b, float m)
	{
		gpu_sigmoid(x.getThrustPointer(), x.size(), a, b, m);
	}
	/**
	 * m * Sigmoid(a * x + b)
	 */
	public static void sigmoid(FloatMat x,  FloatMat out, float a, float b, float m)
	{
		gpu_sigmoid(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b, m);
	}
	public static void sigmoid(FloatMat x) {  sigmoid(x, 1, 0, 1); }
	public static void sigmoid(FloatMat x, FloatMat out) {  sigmoid(x, out, 1, 0, 1); }
	
	/**
	 * m * sigmoid_deriv(a * x + b):  x .* (1 - x)
	 */
	public static void sigmoid_deriv(FloatMat x, float a, float b, float m)
	{
		gpu_sigmoid_deriv(x.getThrustPointer(), x.size(), a, b, m);
	}
	/**
	 * m * sigmoid_deriv(a * x + b):  x .* (1 - x)
	 */
	public static void sigmoid_deriv(FloatMat x,  FloatMat out, float a, float b, float m)
	{
		gpu_sigmoid_deriv(x.getThrustPointer(), x.size(), out.getThrustPointer(), a, b, m);
	}
	public static void sigmoid_deriv(FloatMat x) {  sigmoid_deriv(x, 1, 0, 1); }
	public static void sigmoid_deriv(FloatMat x, FloatMat out) {  sigmoid_deriv(x, out, 1, 0, 1); }
	
	/**
	 * Generate standard Laplacian distribution from uniform i.i.d
	 * Correct for infinity values -> 0
	 */
	public static void laplacian(FloatMat x)
	{
		gpu_laplacian(x.getThrustPointer(), x.size(), 1, 0, 1);
		correct_inf(x);
	}
	public static void laplacian(FloatMat x, FloatMat out)
	{
		gpu_laplacian(x.getThrustPointer(), x.size(), out.getThrustPointer(), 1, 0, 1);
		correct_inf(x);
	}
	
	/**
	 * Generate standard Cauchy distribution from uniform i.i.d
	 * Correct for infinity values -> 0
	 */
	public static void cauchy(FloatMat x)
	{
		gpu_cauchy(x.getThrustPointer(), x.size(), 1, 0, 1);
		correct_inf(x);
	}
	public static void cauchy(FloatMat x, FloatMat out)
	{
		gpu_cauchy(x.getThrustPointer(), x.size(), out.getThrustPointer(), 1, 0, 1);
		correct_inf(x);
	}
	
	/**
	 * Triangular wave, full
	 * @param scale range from -scale to +scale, default = 1
	 * @param halfPeriod [0, halfPeriod), default = PI/2
	 */
	public static void triangular_wave(FloatMat x, float scale, float halfPeriod)
	{
		gpu_triangular_wave(x.getThrustPointer(), x.size(), 1f/halfPeriod, 0, scale);
	}
	public static void trianglar_wave(FloatMat x)
	{
		triangular_wave(x, 1, (float) (Math.PI/2));
	}
	public static void trianglar_wave(FloatMat x, FloatMat out, float scale, float halfPeriod)
	{
		gpu_triangular_wave(x.getThrustPointer(), x.size(), out.getThrustPointer(), 1f/halfPeriod, 0, scale);
	}
	public static void trianglar_wave(FloatMat x, FloatMat out)
	{
		trianglar_wave(x, out, 1, (float) (Math.PI/2));
	}
	
	
	/**
	 * Triangular wave, positive only
	 * @param scale range from 0 to +scale, default = 1
	 * @param halfPeriod [0, halfPeriod), default = PI/2
	 */
	public static void trianglar_wave_positive(FloatMat x, float scale, float halfPeriod)
	{
		gpu_triangular_wave_positive(x.getThrustPointer(), x.size(), 1f/halfPeriod, 0, scale);
	}
	public static void trianglar_wave_positive(FloatMat x)
	{
		trianglar_wave_positive(x, 1, (float) (Math.PI/2));
	}
	public static void trianglar_wave_positive(FloatMat x, FloatMat out, float scale, float halfPeriod)
	{
		gpu_triangular_wave_positive(x.getThrustPointer(), x.size(), out.getThrustPointer(), 1f/halfPeriod, 0, scale);
	}
	public static void trianglar_wave_positive(FloatMat x, FloatMat out)
	{
		trianglar_wave_positive(x, out, 1, (float) (Math.PI/2));
	}

	
	
	public static float sum(FloatMat x)
	{
		return gpu_sum(x.getThrustPointer(), x.size());
	}
	
	public static float product(FloatMat x)
	{
		return gpu_product(x.getThrustPointer(), x.size());
	}
	
	public static float max(FloatMat x)
	{
		return gpu_max(x.getThrustPointer(), x.size());
	}
	
	public static float min(FloatMat x)
	{
		return gpu_min(x.getThrustPointer(), x.size());
	}
	  
    public static float log_sum(FloatMat x)
    {
    	return Natives.gpu_log_sum(x.getThrustPointer(), x.size());
    }

    public static float square_sum(FloatMat x)
    {
    	return Natives.gpu_square_sum(x.getThrustPointer(), x.size());
    }

    public static float abs_sum(FloatMat x)
    {
    	return Natives.gpu_abs_sum(x.getThrustPointer(), x.size());
    }
	
	/**
	 * Element-wise multiplication
	 * @see GpuBlas#dotMult(FloatMat, FloatMat, scalor)
	 */
	public static void dot_mult(FloatMat x, FloatMat y, float scalor)
	{
		gpu_dot_mult(x.getThrustPointer(), x.size(), y.getThrustPointer(), scalor);
	}
	/**
	 * Element-wise multiplication
	 * @see GpuBlas#dotMult(FloatMat, FloatMat, FloatMat)
	 */
	public static void dot_mult(FloatMat x, FloatMat y, FloatMat out, float scalor)
	{
		gpu_dot_mult(x.getThrustPointer(), x.size(), y.getThrustPointer(), out.getThrustPointer(), scalor);
	}
	
	/**
	 * Set a single value
	 */
	public static void set_single(FloatMat x, int idx, float newVal)
	{
		gpu_set_single(x.getThrustPointer(), idx, newVal);
	}
	/**
	 * @param ij can be negative: python wrap-around
	 */
	public static void set_single(FloatMat x, int i, int j, float newVal)
	{
		set_single(x, x.toIndex(i, j), newVal);
	}
	
	/**
	 * Increment a single value
	 */
	public static void incr_single(FloatMat x, int idx, float incrVal)
	{
		gpu_incr_single(x.getThrustPointer(), idx, incrVal);
	}
	/**
	 * @param ij can be negative: python wrap-around
	 */
	public static void incr_single(FloatMat x, int i, int j, float incrVal)
	{
		incr_single(x, x.toIndex(i, j), incrVal);
	}
	
	/**
	 * Sort. dir = 1 for ascending, -1 for descending
	 * Default: ascending
	 */
	public static void sort(FloatMat x, int dir)
	{
		gpu_sort(x.getThrustPointer(), x.size(), dir);
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
		gpu_copy(x.getThrustPointer(), x.size(), out.getThrustPointer());
	}
	
	/**
	 * Swap x with y
	 */
	public static void swap(FloatMat x, FloatMat y)
	{
		gpu_swap(x.getThrustPointer(), x.size(), y.getThrustPointer());
	}
	
	/**
	 * Fill x with val
	 */
	public static void fill(FloatMat x, float val)
	{
		gpu_fill(x.getThrustPointer(), x.size(), val);
	}
	
	 /**
     *  Set a specified row of a column-major matrix to be the same value
     *  @param rowIdx like python, wrapped around: if negative, rowIdx = rowDim + rowIdx
     */
    public static void fill_row(FloatMat x, int rowIdx, float val)
    {
    	Natives.gpu_fill_row(x.getThrustPointer(), x.row, x.col, rowIdx, val);
    }
    /**
     *  Set a specified col of a  column-major matrix to be the same value
     *  @param colIdx like python, wrapped around: if negative, colIdx = colDim + colIdx
     */
    public static void fill_col(FloatMat x, int colIdx, float val)
    {
    	Natives.gpu_fill_col(x.getThrustPointer(), x.row, x.col, colIdx, val);
    }
	
    /**
	 * Actually transpose the matrix data on GPU
	 * NOTE: this isn't the same as the cuBLAS nominal transpose flag!!!
	 * @param x will not be changed in any way
	 * @param out will contain the transposed device data from X
	 */
    public static void transpose(FloatMat x, FloatMat out)
    {
    	if (x == out)
    		throw new GpuException("Transpose operation cannot have the same 'in' and 'out'");
    	Natives.gpu_transpose(x.getThrustPointer(), x.row, x.col, out.getThrustPointer());
    }
    
    /**
     * Thrust version of random normal distr generator
     * The cuRAND one breaks when offset isn't an even integer. 
     */
    public static void fill_rand_normal(FloatMat x, float mean, float stddev)
    {
    	Natives.gpu_fill_rand_normal(x.getThrustPointer(), x.size(), mean, stddev);
    }
    
    /**
     * Correct any infinity values to 0
     */
    public static void correct_inf(FloatMat x)
    {
    	Natives.gpu_correct_inf(x.getThrustPointer(), x.size());
    }

    // ******************** Softmax/labeling methods ****************** /
    // helper for labeling
    public static IntPointer copy_host_to_device(int[] labels)
    {
    	return Natives.copy_host_to_device(new IntPointer(labels), labels.length);
    }
    
    // Set the last row of a matrix to 1
    public static void set_last_row_one(FloatMat x)
    {
    	fill_row(x, -1, 1);
    }
     
    /**
     * Minibatch: softmax(cols)
	 * @param x intrusive: x will be changed unless 'out' is specified
	 * @param hasBias if true, ignore the last row
     */
    public static void batch_softmax(FloatMat x, boolean hasBias)
    {
    	Natives.gpu_batch_softmax(x.getThrustPointer(), x.row, x.col, hasBias);
    }

    public static void batch_softmax(FloatMat x, FloatMat out, boolean hasBias)
    {
    	Natives.gpu_batch_softmax(x.getThrustPointer(), x.row, x.col, out.getThrustPointer(), hasBias);
    }

	/**
	 * Minibatch: softmax(cols) - I[y == j] 
	 * @param x intrusive: x will be changed unless 'out' is specified
	 * @param labels must be already on GPU. Call copy_host_to_device().
	 * @param hasBias if true, ignore the last row
	 */
	public static void batch_softmax_minus_id(FloatMat x, IntPointer labels, boolean hasBias)
	{
		Natives.gpu_batch_softmax_minus_id(x.getThrustPointer(), x.row, x.col, labels, hasBias);
	}
	/**
	 * Minibatch: softmax(cols) - I[y == j] 
	 * @param out result
	 * @param labels must be already on GPU. Call copy_host_to_device().
	 * @param hasBias if true, ignore the last row
	 */
	public static void batch_softmax_minus_id(FloatMat x, FloatMat out, IntPointer labels, boolean hasBias)
	{
		Natives.gpu_batch_softmax_minus_id(x.getThrustPointer(), x.row, x.col, out.getThrustPointer(), labels, hasBias);
	}
	
    /**
     * Minibatch: softmax(alpha_vec)
     * @param x non-intrusive, x won't be changed
     * @param outLogProb writes only the log(prob) at the correct label of a column
	 * @param labels must be already on GPU. Call copy_host_to_device().
	 * @param hasBias if true, ignore the last row
	 * @return sum(outLogProb)
     */
    public static float batch_softmax_at_label(FloatMat x, FloatMat outLogProb, IntPointer labels, boolean hasBias)
    {
    	return Natives.gpu_batch_softmax_at_label(
            			x.getThrustPointer(), x.row, x.col, outLogProb.getThrustPointer(), labels, hasBias);
    }
    
    /**
     * Minibatch: get the labels where the maximum probability occurs
     * @param x non-intrusive, x won't be changed
     * @param reusedDevicePtr use malloc_device() once to malloc on GPU
     * @param outLabels collects the maximum labels, 
     * 				writing from 'offset', write number of labels == label of columns
	 * @param hasBias if true, ignore the last row
     */
    public static void best_label(
    		FloatMat x, IntPointer reusedDevicePtr, int[] outLabels, int offset, boolean hasBias)
	{
    	Natives.gpu_best_label(x.getThrustPointer(), x.row, x.col, reusedDevicePtr, hasBias);
    	Natives.copy_device_to_host(reusedDevicePtr, outLabels, offset, x.col);
	}

   
    // A few duplicates from ThrustNative.java
	// Force Thrust.java to generate code by JavaCpp
    public static native @ByVal FloatDevicePointer offset(@ByVal FloatDevicePointer begin, int offset);
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
