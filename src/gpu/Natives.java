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
    public static native void gpu_exp_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_exp_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

	/**
	 * ln(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_log_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_log_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

	/**
	 * log10(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_log10_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_log10_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

	/**
	 * sqrt(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sqrt_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_sqrt_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

	/**
	 * (a*x + b) ^p in place, or with an output pointer
	 */
    public static native void gpu_pow_float(@ByVal FloatDevicePointer begin, int size, float p, float a, float b);
    public static native void gpu_pow_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float p, float a, float b);
    
    /**
	 * (a*x + b)^2 in place, or with an output pointer
     */
    public static native void gpu_square_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_square_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

    /**
	 * (a*x + b)^3 in place, or with an output pointer
     */
    public static native void gpu_cube_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_cube_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

    /**
	 * 1/(a*x + b) in place, or with an output pointer
     */
    public static native void gpu_reciprocal_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_reciprocal_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

    // trigs
	/**
	 * sin(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sin_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_sin_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
	/**
	 * cos(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_cos_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_cos_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
	/**
	 * tan(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_tan_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_tan_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
    /**
     * a * x + b linear transformation in place, or with an output pointer
     */
    public static native void gpu__float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu__float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

    /**
	 * abs(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_fabs_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_fabs_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
    /**
	 * sigmoid(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sigmoid_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_sigmoid_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
    /**
	 * sigmoid_deriv(a*x + b) <=> x .* (1 - x) in place, or with an output pointer
	 */
    public static native void gpu_sigmoid_deriv_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_sigmoid_deriv_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
    /**
     * Generate Laplacian distribution from a uniform rand i.i.d
     */
    public static native void gpu_laplacian_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_laplacian_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

    /**
     * Generate Cauchy distribution from a uniform rand i.i.d
     */
    public static native void gpu_cauchy_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_cauchy_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
    // other
    public static native float gpu_max_float(@ByVal FloatDevicePointer begin, int size);
    public static native float gpu_min_float(@ByVal FloatDevicePointer begin, int size);

    public static native float gpu_sum_float(@ByVal FloatDevicePointer begin, int size);
    public static native float gpu_product_float(@ByVal FloatDevicePointer begin, int size);
    
    /**
     * Element-wise multiplication
     */
    public static native void gpu_dot_mult_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer begin2, float scalor);
    public static native void gpu_dot_mult_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer begin2, @ByVal FloatDevicePointer out, float scalor);
    
    /**
     * Set a single value 
     */
    public static native void gpu_set_single_float(@ByVal FloatDevicePointer begin, int offset, float newVal);
    /**
     * Increment a single value 
     */
    public static native void gpu_incr_single_float(@ByVal FloatDevicePointer begin, int offset, float incrVal);
    
    
    /**
     * Sort: dir = 1 ascending, dir = -1 descending
     */
    public static native void gpu_sort_float(@ByVal FloatDevicePointer begin, int size, int dir);

    /**
     * Copy to 'out'
     */
    public static native void gpu_copy_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out);
    
    /**
     * Swap this with 'out'
     */
    public static native void gpu_swap_float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out);
    
    /**
     * Fill the GPU array with 'val'
     */
    public static native void gpu_fill_float(@ByVal FloatDevicePointer begin, int size, float val);
    
    /**
     *  Utility: pointer += offset, advance the GPU pointer
     */
    public static native @ByVal FloatDevicePointer offset(@ByVal FloatDevicePointer begin, int offset);

    /**
     *  Set a specified row of a column-major matrix to be the same value
     */
    public static native void gpu_fill_row_float(
    		@ByVal FloatDevicePointer begin, int row, int col, int rowIdx, float val);
    /**
     *  Set a specified col of a  column-major matrix to be the same value
     */
    public static native void gpu_fill_col_float(
    		@ByVal FloatDevicePointer begin, int row, int col, int colIdx, float val);
    
    /**
     * Transpose the actual data matrix on GPU
     */
    public static native void gpu_transpose_float(
    		@ByVal FloatDevicePointer in, int row, int col, @ByVal FloatDevicePointer out);
    
    
    // ******************** Softmax/labeling specific methods ****************** /
    /**
     * I[y == j] - softmax(alpha_vec)
     */
    public static native void gpu_id_minus_softmax(@ByVal FloatDevicePointer begin, int size, int id);
    // version 2: more calculation, might be more numerically stable
    public static native void gpu_id_minus_softmax_2(@ByVal FloatDevicePointer begin, int size, int id);
    
    // For minibatch
    public static native void gpu_batch_id_minus_softmax(
    		@ByVal FloatDevicePointer begin, int row, int col, @ByPtr IntPointer labels);
    
    // To calculate softmax() only, no subtraction from id[]
    public static native void gpu_batch_softmax(@ByVal FloatDevicePointer begin, int row, int col);

    public static native void gpu_batch_softmax(@ByVal FloatDevicePointer begin, int row, int col, @ByVal FloatDevicePointer out);

    // Only the probability at the correct label
    public static native void gpu_batch_softmax(
    				@ByVal FloatDevicePointer begin, int row, int col, 
    				@ByVal FloatDevicePointer out, @ByPtr IntPointer labels);

    // The best labels
    public static native void gpu_best_label(
    		@ByVal FloatDevicePointer begin, int row, int col, @ByPtr IntPointer outLabels);
    
    // Sum of log probability from the correct label
    public static native float gpu_log_sum(@ByVal FloatDevicePointer begin, int size);
    
    // combine babel_batch_id_minus_softmax with babel_log_prob
    public static native float gpu_batch_id_minus_softmax_log_prob(
    		@ByVal FloatDevicePointer begin, int row, int col, 
    		@ByVal FloatDevicePointer outLogProb, @ByPtr IntPointer labels);
    
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
    public static native void gpu_exp_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_exp_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

	/**
	 * ln(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_log_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_log_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

	/**
	 * log10(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_log10_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_log10_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

	/**
	 * sqrt(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sqrt_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_sqrt_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

	/**
	 * (a*x + b) ^p in place, or with an output pointer
	 */
    public static native void gpu_pow_double(@ByVal DoubleDevicePointer begin, int size, double p, double a, double b);
    public static native void gpu_pow_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double p, double a, double b);

    // trigs
	/**
	 * sin(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sin_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_sin_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);
    
	/**
	 * cos(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_cos_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_cos_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);
    
	/**
	 * tan(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_tan_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_tan_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);
    
    /**
     * a * x + b linear transformation
     */
    public static native void gpu__double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu__double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    /**
     * abs(a*x + b) in place, or with an output pointer
     */
    public static native void gpu_fabs_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_fabs_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    /**
     * sigmoid(a*x + b) in place, or with an output pointer
     */
    public static native void gpu_sigmoid_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_sigmoid_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    /**
     * sigmoid_deriv(a*x + b) <=> x .* (1 - x) in place, or with an output pointer
     */
    public static native void gpu_sigmoid_deriv_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_sigmoid_deriv_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    /**
     * Generate Laplacian distribution from a uniform rand i.i.d
     */
    public static native void gpu_laplacian_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_laplacian_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    /**
     * Generate Cauchy distribution from a uniform rand i.i.d
     */
    public static native void gpu_cauchy_double(@ByVal DoubleDevicePointer begin, int size, double a, double b);
    public static native void gpu_cauchy_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out, double a, double b);

    // other
    public static native double gpu_max_double(@ByVal DoubleDevicePointer begin, int size);
    public static native double gpu_min_double(@ByVal DoubleDevicePointer begin, int size);

    public static native double gpu_sum_double(@ByVal DoubleDevicePointer begin, int size);
    public static native double gpu_product_double(@ByVal DoubleDevicePointer begin, int size);

    /**
     * Element-wise multiplication
     */
    public static native void gpu_dot_mult_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer begin2, double scalor);
    public static native void gpu_dot_mult_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer begin2, @ByVal DoubleDevicePointer out, double scalor);
    
    /**
     * Set a single value 
     */
    public static native void gpu_set_single_double(@ByVal DoubleDevicePointer begin, int offset, double newVal);
    /**
     * Increment a single value 
     */
    public static native void gpu_incr_single_double(@ByVal DoubleDevicePointer begin, int offset, double incrVal);
    
    /**
     * Sort: dir = 1 ascending, dir = -1 descending
     */
    public static native void gpu_sort_double(@ByVal DoubleDevicePointer begin, int size, int dir);

    /**
     * Copy to 'out'
     */
    public static native void gpu_copy_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out);
    
    /**
     * Swap this with 'out'
     */
    public static native void gpu_swap_double(@ByVal DoubleDevicePointer begin, int size, @ByVal DoubleDevicePointer out);
    
    /**
     * Fill the GPU array with 'val'
     */
    public static native void gpu_fill_double(@ByVal DoubleDevicePointer begin, int size, double val);
    
    /**
     *  Utility: pointer += offset, advance the GPU pointer
     */
    public static native @ByVal DoubleDevicePointer offset(@ByVal DoubleDevicePointer begin, int offset);

    
    // ******************** Babel specific methods ****************** /
    /**
     * I[y == j] - softmax(alpha_vec)
     */
    public static native void gpu_id_minus_softmax(@ByVal DoubleDevicePointer begin, int size, int id);
}
