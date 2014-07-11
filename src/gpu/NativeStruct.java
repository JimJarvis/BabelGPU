package gpu;

import com.googlecode.javacpp.*;
import com.googlecode.javacpp.annotation.*;

@Platform(include={"<thrust/host_vector.h>", "<thrust/device_vector.h>", "<thrust/generate.h>", "<thrust/sort.h>",
                   "<thrust/copy.h>", "<thrust/reduce.h>", "<thrust/functional.h>", "<algorithm>", "<cstdlib>"})
@Namespace("thrust")
/**
 * Natives structures to connect to the Thrust API
 */
public class NativeStruct
{
	static { Loader.load(); }
	
	/**
	 * Most important struct to talk to the Thrust API
	 * Can convert directly from jcuda.Pointer
	 */
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
        public FloatDevicePointer offset(int n)
        {		
        	return Natives.offset(this, n);
        }
        
        private native void allocate(FloatPointer ptr);
        public native FloatPointer get();
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
    
    //**************************************************/
	//******************* DOUBLE *******************/
	//**************************************************/
    /**
	 * Most important struct to talk to the Thrust API
	 * Can convert directly from jcuda.Pointer
	 */
	@Name("device_ptr<double>")
	public static class DoubleDevicePointer extends Pointer
	{
        static { Loader.load(); }
        public DoubleDevicePointer() { allocate(null); }
        public DoubleDevicePointer(DoublePointer ptr) { allocate(ptr); }
        public DoubleDevicePointer(final jcuda.Pointer p)
        { 
        	this(new DoublePointer((DoublePointer)null)
        	{ 
        		{ 
        			address = new jcuda.Pointer(p) 
        			{ 
        				long a = getNativePointer();
        			}.a;
        		} 
        	}); 
        }
        public DoubleDevicePointer offset(int n)
        {		
        	return Natives.offset(this, n);
        }
        
        private native void allocate(DoublePointer ptr);
        public native DoublePointer get();
    }
    
	@Name("host_vector<double>")
    public static class DoubleHostVector extends Pointer
    {
        static { Loader.load(); }
        public DoubleHostVector() { allocate(0); }
        public DoubleHostVector(long n) { allocate(n); }
        public DoubleHostVector(DoubleDeviceVector v) { allocate(v); }
        private native void allocate(long n);
        private native void allocate(@ByRef DoubleDeviceVector v);

        public DoublePointer begin() { return data(); }
        public DoublePointer end() { return data().position((int)size()); }

        public native DoublePointer data();
        public native long size();
        public native void resize(long n);
    }
	
    
    @Name("device_vector<double>")
    public static class DoubleDeviceVector extends Pointer
    {
        static { Loader.load(); }
        public DoubleDeviceVector() { allocate(0); }
        public DoubleDeviceVector(long n) { allocate(n); }
        public DoubleDeviceVector(DoubleHostVector v) { allocate(v); }
        private native void allocate(long n);
        private native void allocate(@ByRef DoubleHostVector v);

        public DoubleDevicePointer begin() { return data(); }
        public DoubleDevicePointer end() { return new DoubleDevicePointer(data().get().position((int)size())); }

        public native @ByVal DoubleDevicePointer data();
        public native long size();
        public native void resize(long n);
    }
}
