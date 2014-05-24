package gpu;

import com.googlecode.javacpp.*;
import com.googlecode.javacpp.annotation.*;

@Platform(include={"<thrust/host_vector.h>", "<thrust/device_vector.h>", "<thrust/generate.h>", "<thrust/sort.h>",
                   "<thrust/copy.h>", "<thrust/reduce.h>", "<thrust/functional.h>", "<algorithm>", "<cstdlib>"})
@Namespace("thrust")
public class Thrust
{
	static { Loader.load(); }
	
	@Name("plus<float>")
    public static class FloatPlus extends Pointer {
        static { Loader.load(); }
        public FloatPlus() { allocate(); }
        private native void allocate();
        public native @Name("operator()") int call(int x, int y);
    }
	
	@Name("host_vector<float>")
    public static class FloatHostVector extends Pointer
    {
        static { Loader.load(); }
        public FloatHostVector() { allocate(0); }
        public FloatHostVector(long n) { allocate(n); }
        public FloatHostVector(FloatDeviceVector v) { allocate(v); }
        private native void allocate(long n);
        private native void allocate(@ByRef FloatDeviceVector v);

        public FloatPointer begin() { return data(); }
        public FloatPointer end() { return data().position((int)size()); }

        public native FloatPointer data();
        public native long size();
        public native void resize(long n);
    }
	
    @Name("device_ptr<float>")
    public static class FloatDevicePointer extends Pointer
    {
        static { Loader.load(); }
        public FloatDevicePointer() { allocate(null); }
        public FloatDevicePointer(FloatPointer ptr) { allocate(ptr); }
        public FloatDevicePointer(final jcuda.Pointer p)
        { 
        	this(new FloatPointer((FloatPointer)null)
        	{ 
        		{ 
        			address = new jcuda.Pointer(p) 
        			{ 
        				long a = getNativePointer();
        			}.a;
        		} 
        	}); 
        } 
        
        private native void allocate(FloatPointer ptr);
        public native FloatPointer get();
    }
    
    @Name("device_vector<float>")
    public static class FloatDeviceVector extends Pointer
    {
        static { Loader.load(); }
        public FloatDeviceVector() { allocate(0); }
        public FloatDeviceVector(long n) { allocate(n); }
        public FloatDeviceVector(FloatHostVector v) { allocate(v); }
        private native void allocate(long n);
        private native void allocate(@ByRef FloatHostVector v);

        public FloatDevicePointer begin() { return data(); }
        public FloatDevicePointer end() { return new FloatDevicePointer(data().get().position((int)size())); }

        public native @ByVal FloatDevicePointer data();
        public native long size();
        public native void resize(long n);
    }
    
    public static native void sort(@ByVal FloatDevicePointer first, @ByVal FloatDevicePointer last);
    public static native float reduce(@ByVal FloatDevicePointer first, @ByVal FloatDevicePointer last, int init, @ByVal FloatPlus binary_op);
    
    
    public static void main(String[] args)
	{
		
	}
}
