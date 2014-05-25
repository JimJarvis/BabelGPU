package gpu;
import gpu.ThrustStruct.*;

import com.googlecode.javacpp.*;
import com.googlecode.javacpp.annotation.*;

@Platform(include={"\"my_gpu.h\"", "\"babel_gpu.h\""})
@Namespace("MyGpu")
/**
 * Interface to Thrust functions in my_gpu.h
 */
public class Thrust
{
	static { Loader.load(); }
	
    public static native float gpu_max_float(@ByVal FloatDevicePointer begin, int size);
    public static native void gpu_exp_float(@ByVal FloatDevicePointer begin, int size, float a, float b);
    public static native float gpu_sum_float(@ByVal FloatDevicePointer begin, int size);
    public static native void babel_id_minus_softmax(@ByVal FloatDevicePointer begin, int size, int id);
    
    // Utility: easier pointer manipulation
    public static native @ByVal FloatDevicePointer offset_float(@ByVal FloatDevicePointer begin, int offset);
}
