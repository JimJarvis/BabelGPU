package gpu;
import gpu.ThrustStruct.*;

import com.googlecode.javacpp.*;
import com.googlecode.javacpp.annotation.*;

@Platform(include={"\"my_gpu.h\"", "\"babel_gpu.h\""})
@Namespace("MyGpu")
/**
 * Interface to Thrust functions in my_gpu.h
 */
public class ThrustNative
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
     * a * x + b linear transformation
     */
    public static native void gpu__float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native void gpu__float(@ByVal FloatDevicePointer begin, int size, @ByVal FloatDevicePointer out, float a, float b);
    
    // other
    public static native float gpu_max_float(@ByVal FloatDevicePointer begin, int size);
    public static native float gpu_min_float(@ByVal FloatDevicePointer begin, int size);

    public static native float gpu_sum_float(@ByVal FloatDevicePointer begin, int size);
    public static native float gpu_product_float(@ByVal FloatDevicePointer begin, int size);
    
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
    public static native @ByVal FloatDevicePointer offset_float(@ByVal FloatDevicePointer begin, int offset);

    
    // ******************** Babel specific methods ****************** /
    /**
     * I[y == j] - softmax(alpha_vec)
     */
    public static native void babel_id_minus_softmax(@ByVal FloatDevicePointer begin, int size, int id);
    // version 2: more calculation, might be more numerically stable
    public static native void babel_id_minus_softmax_2(@ByVal FloatDevicePointer begin, int size, int id);
    
}
