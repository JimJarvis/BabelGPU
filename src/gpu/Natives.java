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
	 * m * exp(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_exp(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_exp(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);

	/**
	 * m * ln(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_log(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_log(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);

	/**
	 * m * log10(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_log10(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_log10(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);

	/**
	 * m * sqrt(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sqrt(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_sqrt(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);

	/**
	 * m * (a*x + b) ^p in place, or with an output pointer
	 */
    public static native void gpu_pow(@ByVal FloatDevicePointer begin, int size, float p, float a, float b, float m);
    public static native void gpu_pow(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float p, float a, float b, float m);
    
    /**
	 * m * (a*x + b)^2 in place, or with an output pointer
     */
    public static native void gpu_square(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_square(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);

    /**
	 * m * (a*x + b)^3 in place, or with an output pointer
     */
    public static native void gpu_cube(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_cube(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);

    /**
	 * m * 1/(a*x + b) in place, or with an output pointer
     */
    public static native void gpu_reciprocal(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_reciprocal(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);

    // trigs
	/**
	 * m * sin(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sin(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_sin(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);
    
	/**
	 * m * cos(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_cos(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_cos(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);
    
	/**
	 * m * tan(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_tan(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_tan(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);
    
    /**
     * a * x + b linear transformation in place, or with an output pointer
     */
    public static native void gpu_linear(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu_linear(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);

    /**
	 * m * abs(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_fabs(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_fabs(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);
    
    /**
	 * m * sigmoid(a*x + b) in place, or with an output pointer
	 */
    public static native void gpu_sigmoid(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_sigmoid(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);
    
    /**
	 * m * sigmoid_deriv(a*x + b) <=> x .* (1 - x) in place, or with an output pointer
	 */
    public static native void gpu_sigmoid_deriv(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_sigmoid_deriv(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);
    
    /**
     * Generate Laplacian distribution from a uniform rand i.i.d
     */
    public static native void gpu_laplacian(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_laplacian(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);

    /**
     * Generate Cauchy distribution from a uniform rand i.i.d
     */
    public static native void gpu_cauchy(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_cauchy(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);
    
    /**
     * Triangular wave
     */
    public static native void gpu_triangular_wave(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_triangular_wave(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);
    
    /**
     * Triangular wave positive only
     */
    public static native void gpu_triangular_wave_positive(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_triangular_wave_positive(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);
    
    /**
     * Rectified linear function: x > 0 ? x : 0
     */
    public static native void gpu_rectified_linear(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_rectified_linear(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);
    
    /**
     * Rectified linear function: x > 0 ? 1 : 0
     */
    public static native void gpu_rectified_linear_deriv(@ByVal FloatDevicePointer begin, int size, float a, float b, float m);
    public static native void gpu_rectified_linear_deriv(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b, float m);
    
    /***** other ******/
    public static native float gpu_max(@ByVal FloatDevicePointer begin, int size);
    public static native float gpu_min(@ByVal FloatDevicePointer begin, int size);

    public static native float gpu_sum(@ByVal FloatDevicePointer begin, int size);
    public static native float gpu_product(@ByVal FloatDevicePointer begin, int size);
    
    // Sum of log(x)
    public static native float gpu_log_sum(@ByVal FloatDevicePointer begin, int size);
    // Sum of x^2
    public static native float gpu_square_sum(@ByVal FloatDevicePointer begin, int size);
    // Sum of |x|
    public static native float gpu_abs_sum(@ByVal FloatDevicePointer begin, int size);
    
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
    
    /**
     * Fill with random gaussian distribution
     */
    public static native void gpu_fill_rand_normal(
    		@ByVal FloatDevicePointer begin, int size, float mean, float stddev);
    
    /**
     * Correct any infinity values to 0
     */
    public static native void gpu_correct_inf(@ByVal FloatDevicePointer begin, int size);
    
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
    /**
     *  Utility: pointer += offset, advance the GPU pointer
     */
    public static native @ByVal DoubleDevicePointer offset(@ByVal DoubleDevicePointer begin, int offset);
}
