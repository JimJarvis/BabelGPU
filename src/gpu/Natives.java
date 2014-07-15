package gpu;
import gpu.NativeStruct.*;

import com.googlecode.javacpp.*;
import com.googlecode.javacpp.annotation.*;

@Platform(include={"\"my_gpu.h\"", "\"my_kernel.h\""})
@Namespace("MyGpu")
/**
 * Interface to Thrust functions in my_gpu.h
 */
public class Natives
{
	static { Loader.load(); }
	
	// Exp and logs
	/**
	 * exp(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_exp(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_exp(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

	/**
	 * ln(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_log(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_log(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

	/**
	 * log10(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_log10(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_log10(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

	/**
	 * sqrt(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sqrt(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_sqrt(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

	/**
	 * (a*x + b) ^p in place, or with an output pointer
	 */
    public static native void gpu_pow(@ByVal FloatDevicePointer begin, int size, float p, float a, float b);
    public static native void gpu_pow(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float p, float a, float b);
    
    /**
	 * (a*x + b)^2 in place, or with an output pointer
     */
    public static native void gpu_square(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_square(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

    /**
	 * (a*x + b)^3 in place, or with an output pointer
     */
    public static native void gpu_cube(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_cube(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

    /**
	 * 1/(a*x + b) in place, or with an output pointer
     */
    public static native void gpu_reciprocal(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_reciprocal(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

    // trigs
	/**
	 * sin(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sin(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_sin(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
	/**
	 * cos(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_cos(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_cos(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
	/**
	 * tan(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_tan(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_tan(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
    /**
     * a * x + b linear transformation in place, or with an output pointer
     */
    public static native void gpu_(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

    /**
	 * abs(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_fabs(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_fabs(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
    /**
	 * sigmoid(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sigmoid(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_sigmoid(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
    /**
	 * sigmoid_deriv(a*x + b) <=> x .* (1 - x) in place, or with an output pointer
	 */
    public static native void gpu_sigmoid_deriv(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_sigmoid_deriv(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
    /**
     * Generate Laplacian distribution from a uniform rand i.i.d
     */
    public static native void gpu_laplacian(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_laplacian(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

    /**
     * Generate Cauchy distribution from a uniform rand i.i.d
     */
    public static native void gpu_cauchy(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_cauchy(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
    // other
    public static native float gpu_max(@ByVal FloatDevicePointer begin, int size);
    public static native float gpu_min(@ByVal FloatDevicePointer begin, int size);

    public static native float gpu_sum(@ByVal FloatDevicePointer begin, int size);
    public static native float gpu_product(@ByVal FloatDevicePointer begin, int size);
    
    // Sum of log(x)
    public static native float gpu_log_sum(@ByVal FloatDevicePointer begin, int size);
    
    /**
     * Element-wise multiplication
     */
    public static native void gpu_dot_mult(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer begin2, float scalor);
    public static native void gpu_dot_mult(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer begin2, @ByVal FloatDevicePointer out, float scalor);
    
    /**
     * Set a single value 
     */
    public static native void gpu_set_single(@ByVal FloatDevicePointer begin, int offset, float newVal);
    /**
     * Increment a single value 
     */
    public static native void gpu_incr_single(@ByVal FloatDevicePointer begin, int offset, float incrVal);
    
    
    /**
     * Sort: dir = 1 ascending, dir = -1 descending
     */
    public static native void gpu_sort(@ByVal FloatDevicePointer begin, int size, int dir);

    /**
     * Copy to 'out'
     */
    public static native void gpu_copy(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out);
    
    /**
     * Swap this with 'out'
     */
    public static native void gpu_swap(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out);
    
    /**
     * Fill the GPU array with 'val'
     */
    public static native void gpu_fill(@ByVal FloatDevicePointer begin, int size, float val);
    
    /**
     *  Utility: pointer += offset, advance the GPU pointer
     */
    public static native @ByVal FloatDevicePointer offset(@ByVal FloatDevicePointer begin, int offset);

    /**
     *  Set a specified row of a column-major matrix to be the same value
     */
    public static native void gpu_fill_row(
    		@ByVal FloatDevicePointer begin, int row, int col, int rowIdx, float val);
    /**
     *  Set a specified col of a  column-major matrix to be the same value
     */
    public static native void gpu_fill_col(
    		@ByVal FloatDevicePointer begin, int row, int col, int colIdx, float val);
    
    /**
     * Transpose the actual data matrix on GPU
     */
    public static native void gpu_transpose(
    		@ByVal FloatDevicePointer in, int row, int col, @ByVal FloatDevicePointer out);
    
    
    // ******************** Softmax/labeling specific methods ****************** /
    /**
     * To calculate softmax() only, no subtraction from id[]
     * intrusive: changes input data unless 'out' is specified
     */
    public static native void gpu_batch_softmax(@ByVal FloatDevicePointer begin, int row, int col, boolean hasBias);
    public static native void gpu_batch_softmax(@ByVal FloatDevicePointer begin, int row, int col, @ByVal FloatDevicePointer out, boolean hasBias);
    
    /**
     *  Only the probability at the correct label
     *  Non-intrusive: 'input' won't be changed
     *  @return sum(outLogProb)
     */
    public static native float gpu_batch_softmax_at_label(
    				@ByVal FloatDevicePointer begin, int row, int col, 
    				@ByVal FloatDevicePointer outLogProb, @ByPtr IntPointer labels, boolean hasBias);
    
    /**
     * softmax(alpha_vec) - I[y == j]
     * Uses Thrust, only 1 col
     */
    public static native void gpu_batch_softmax_minus_id(
    		@ByVal FloatDevicePointer begin, int row, int col, @ByPtr IntPointer labels, boolean hasBias);
    public static native void gpu_batch_softmax_minus_id(
    		@ByVal FloatDevicePointer begin, int row, int col, @ByVal FloatDevicePointer out, @ByPtr IntPointer labels, boolean hasBias);
    
    // The best labels, non-intrusive
    public static native void gpu_best_label(
    		@ByVal FloatDevicePointer begin, int row, int col, @ByPtr IntPointer outLabels, boolean hasBias);
    
    // Helper for minibatch
    public static native @ByPtr IntPointer copy_host_to_device(@ByPtr IntPointer host, int size);
    public static native @ByPtr IntPointer copy_device_to_host(@ByPtr IntPointer device, int size);
    // NOTE: @Ptr can directly map to java primitive array types!!!!
    public static native void copy_device_to_host(@ByPtr IntPointer device, @ByPtr int[] host, int offset, int size);
    
    public static native @ByPtr IntPointer malloc_device_int(int size, boolean memsetTo0);
    public static native @ByPtr FloatPointer malloc_device_float(int size, boolean memsetTo0);
    public static native void free_device(@ByPtr IntPointer device);
    public static native void free_host(@ByPtr IntPointer host);
	public static native @ByPtr IntPointer offset(@ByPtr IntPointer begin, int offset);
	
    
    //**************************************************/
	//******************* DOUBLE *******************/
	//**************************************************/
	// Exp and logs
	/**
	 * exp(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_exp(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_exp(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

	/**
	 * ln(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_log(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_log(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

	/**
	 * log10(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_log10(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_log10(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

	/**
	 * sqrt(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sqrt(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_sqrt(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

	/**
	 * (a*x + b) ^p in place, or with an output pointer
	 */
    public static native void gpu_pow(@ByVal DoubleDevicePointer begin, int size, double p, double a, double b);
    public static native void gpu_pow(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double p, double a, double b);

    // trigs
	/**
	 * sin(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sin(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_sin(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);
    
	/**
	 * cos(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_cos(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_cos(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);
    
	/**
	 * tan(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_tan(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_tan(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);
    
    /**
     * a * x + b linear transformation
     */
    public static native void gpu_(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    /**
     * abs(a*x + b) in place, or with an output pointer
     */
    public static native void gpu_fabs(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_fabs(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    /**
     * sigmoid(a*x + b) in place, or with an output pointer
     */
    public static native void gpu_sigmoid(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_sigmoid(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    /**
     * sigmoid_deriv(a*x + b) <=> x .* (1 - x) in place, or with an output pointer
     */
    public static native void gpu_sigmoid_deriv(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_sigmoid_deriv(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    /**
     * Generate Laplacian distribution from a uniform rand i.i.d
     */
    public static native void gpu_laplacian(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_laplacian(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    /**
     * Generate Cauchy distribution from a uniform rand i.i.d
     */
    public static native void gpu_cauchy(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_cauchy(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    // other
    public static native double gpu_max(@ByVal DoubleDevicePointer begin, int size);
    public static native double gpu_min(@ByVal DoubleDevicePointer begin, int size);

    public static native double gpu_sum(@ByVal DoubleDevicePointer begin, int size);
    public static native double gpu_product(@ByVal DoubleDevicePointer begin, int size);

    /**
     * Element-wise multiplication
     */
    public static native void gpu_dot_mult(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer begin2, double scalor);
    public static native void gpu_dot_mult(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer begin2, @ByVal DoubleDevicePointer out, double scalor);
    
    /**
     * Set a single value 
     */
    public static native void gpu_set_single(@ByVal DoubleDevicePointer begin, int offset, double newVal);
    /**
     * Increment a single value 
     */
    public static native void gpu_incr_single(@ByVal DoubleDevicePointer begin, int offset, double incrVal);
    
    /**
     * Sort: dir = 1 ascending, dir = -1 descending
     */
    public static native void gpu_sort(@ByVal DoubleDevicePointer begin, int size, int dir);

    /**
     * Copy to 'out'
     */
    public static native void gpu_copy(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out);
    
    /**
     * Swap this with 'out'
     */
    public static native void gpu_swap(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out);
    
    /**
     * Fill the GPU array with 'val'
     */
    public static native void gpu_fill(@ByVal DoubleDevicePointer begin, int size, double val);
    
    /**
     *  Utility: pointer += offset, advance the GPU pointer
     */
    public static native @ByVal DoubleDevicePointer offset(@ByVal DoubleDevicePointer begin, int offset);
}
