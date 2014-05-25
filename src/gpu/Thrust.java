package gpu;
import gpu.ThrustStruct.*;

import com.googlecode.javacpp.*;
import com.googlecode.javacpp.annotation.*;

@Platform(include={"\"my_gpu.h\"", "\"babel_gpu.h\""})
@Namespace("MyGpu")
public class Thrust
{
	static { Loader.load(); }
	
    public static native float gpu_max_float(@ByVal FloatDevicePointer first, @ByVal FloatDevicePointer last);
    public static native void gpu_exp_float(@ByVal FloatDevicePointer first, @ByVal FloatDevicePointer last, float a, float b);
    public static native int babel_id_minus_softmax(@ByVal FloatDevicePointer first, @ByVal FloatDevicePointer last, int id);
}
