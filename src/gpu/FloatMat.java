package gpu;

import java.nio.FloatBuffer;

import gpu.ThrustStruct.FloatDevicePointer;
import utils.CpuUtil.Coord;
import utils.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.runtime.JCuda.*;
import static jcuda.jcublas.cublasOperation.*;

/**
 * Struct around a matrix with row/col dimension info
 * HostMode: either hostArray or hostBuffer. hostArray has priority.
 */
public class FloatMat
{
	// Should only use either hostArray or hostBuf at a time
	private float[] hostArray = null;
	private FloatBuffer hostBuffer = null;
	/**
	 * None: everything's on device
	 * Array: float array on host
	 * Buffer: FloatBuffer on host
	 */
	public static enum HostMode {None, Array, Buffer};
	private HostMode hostMode = HostMode.None;

	private Pointer device = null; // jCuda pointer
	private FloatDevicePointer thrustPtr = null; // Thrust pointer
	
	// This field records whether the matrix should be transposed or not
	private int op = CUBLAS_OP_N; 
	public int row;
	public int col;
	// Leading dimension: column length (i.e. row dim)
	// Doesn't change even with transpose
	public int ldim; 
	
	private static boolean hostModeCheck = true; // see enableHostModeCheck() method
	
	public static final FloatMat DUMMY = new FloatMat();
	
	/**
	 * Default ctor
	 */
	public FloatMat() {	}
	
	/**
	 * Ctor from host array
	 */
	public FloatMat(float[] host, int row, int col)
	{
		this.hostArray = host;
		this.hostMode = HostMode.Array;
		initDim(row, col);
	}
	
	/**
	 * Ctor from 2D host array
	 */
	public FloatMat(float[][] host)
	{
		this(flatten(host), host.length, host[0].length);
	}
	
	/**
	 * Ctor for 1D vector-array (column vector)
	 */
	public FloatMat(float[] host)
	{
		this(host, host.length, 1);
	}
	
	/**
	 * Ctor from 2D host buffer
	 */
	public FloatMat(FloatBuffer host, int row, int col)
	{
		this.hostBuffer = host;
		this.hostMode = HostMode.Buffer;
		initDim(row, col);
	}
	
	/**
	 * Ctor from 1D vector-buffer
	 */
	public FloatMat(FloatBuffer host, int len)
	{
		this(host, len, 1);
	}

	/**
	 * Ctor from device pointer
	 */
	public FloatMat(Pointer device, int row, int col)
	{
		this.device = device;
		initDim(row, col);
	}
	
	/**
	 * Ctor from device pointer: 1D vector (column vector)
	 */
	public FloatMat(Pointer device, int len)
	{
		this(device, len, 1);
	}
	
	/**
	 * Ctor with dimensions
	 * @param memsetToZero true to initialize the device data to 0. Default true. 
	 */
	public FloatMat(int row, int col, boolean memsetToZero)
	{
		this.device = GpuUtil.allocDeviceFloat(row * col, memsetToZero);
		initDim(row, col);
	}
	
	/**
	 * Ctor with dimensions
	 * The device data will be initialized to all 0
	 */
	public FloatMat(int row, int col)
	{
		this(row, col, true);
	}
	
	/**
	 * Instantiate a new empty FloatMat with the same size
	 * NOTE: doesn't copy any data. Only the same row/col
	 */
	public FloatMat(FloatMat other)
	{
		this(other.row, other.col);
	}
	
	// Ctor helper
	private void initDim(int row, int col)
	{
		this.row = row;
		this.col = col;
		this.ldim = row;
	}
	
	// Shallow copy create new instance
	private FloatMat shallowCopy()
	{
		FloatMat mat = new FloatMat();
		mat.row = this.row;
		mat.col = this.col;
		mat.ldim = this.ldim;
		mat.op = this.op;
		mat.device = this.device;
		mat.hostArray = this.hostArray;
		mat.hostBuffer = this.hostBuffer;
		mat.hostMode = this.hostMode;
		mat.thrustPtr = this.thrustPtr;
		
		return mat;
	}
	
	/**
	 * ONLY clones the device data (GPU copy)
	 * Use with caution! Make sure to free the new device memory!!!
	 * @param new FloatMat with new device memory
	 */
	@Override
	public FloatMat clone()
	{
		if (this.device == null)
			throw new GpuException("Device is null, cannot clone.");
		FloatMat clone = new FloatMat(this);
		clone.copyFrom(this);
		return clone;
	}
	
	/**
	 * Transpose the matrix and return a new one
	 * Nothing in the real data actually changes, but only a flag
	 * @return new instance
	 */
	public FloatMat transpose()
	{
		// Swap row and col dimension
		FloatMat mat = this.shallowCopy();
		mat.row = this.col;
		mat.col = this.row;
		mat.op = (op != CUBLAS_OP_N) ? 
				CUBLAS_OP_N : CUBLAS_OP_T;
		return mat;
	}
	
	public int getOp() {	return this.op;	}
	
	/**
	 * Invariant to transpose
	 */
	public int getOriginalRow()
	{
		return op == CUBLAS_OP_N ? row : col;
	}

	/**
	 * Invariant to transpose
	 */
	public int getOriginalCol()
	{
		return op == CUBLAS_OP_N ? col : row;
	}
	
	/**
	 * Set the memory of the device pointer to 0
	 */
	public void clearDevice()
	{
		if (device == null)
			throw new GpuException("Device is null, cannot clear");
		GpuUtil.clearDeviceFloat(device, size());
	}
	
	/**
	 * Not recommended.
	 * Manually set the HostMode
	 */
	public void setHostMode(HostMode hostMode)
	{
		this.hostMode = hostMode;
	}
	
	/**
	 * Copy to and return the device pointer.
	 * If current HostMode is None, 
	 * we take whichever hostArray or hostBuffer that isn't null
	 * hostArray is given priority. 
	 * Then HostMode is set to the one that isn't null
	 * @param forceCopy default: false, only copy if device pointer is null
	 * otherwise retrieve device without copying.
	 * true: copy to device no matter what
	 */
	public Pointer toDevice(boolean forceCopy)
	{
		if (device != null && !forceCopy)
			return device; // retrieve without copying
		
		if (hostArray == null && hostBuffer == null)
			throw new GpuException("At least one of hostArray and hostBuffer should not be null");
		
		boolean useArray = true;  // default: get device from hostArray
		if (hostMode == HostMode.None) 
		{
			useArray = hostArray != null;
			hostMode = useArray ? HostMode.Array : HostMode.Buffer;
		}
		else
			useArray = hostMode == HostMode.Array;
		
    	if (device == null)
    		device = GpuUtil.allocDeviceFloat(size());
    	if (useArray)
    		GpuUtil.hostToDeviceFloat(hostArray, device, size());
    	else
    		GpuUtil.hostToDeviceFloat(hostBuffer, device, size());
		return device;
	}
	
	/**
	 * Copy to and return device pointer
	 * @see FloatMat#toDevice(boolean) toDevice(false)
	 */
	public Pointer toDevice() {	return toDevice(false);	}
	
	/**
	 * Class-wide setting
	 * If strict check is on, you can only use one channel. 
	 * Either array or buffer, but not both.
	 * For example, you can't call toHostArray and toHostBuffer successively
	 * @param hostModeCheck default true
	 */
	public static void enableHostModeCheck(boolean hostModeCheck)
	{
		FloatMat.hostModeCheck = hostModeCheck;
	}
	
	// private helper, used in to/setHostArray/Buffer
	// can be disabled by enableHostModeCheck(false)
	private void checkHostMode(HostMode correctMode)
	{
		if (!hostModeCheck || hostMode == HostMode.None)
			hostMode = correctMode;
		else if (hostMode != correctMode)
			throw new GpuException(
					String.format("HostMode conflict: attempt to get %s while HostMode is %s", 
							correctMode.toString(), hostMode.toString()));
	}

	/**
	 * Copy to and return host array
	 * Conflict exception if HostMode is Buffer
	 * If HostMode is None, set to Array
	 * If device is null, return current hostArray
	 * @param forceCopy default false: if hostArray already has a value, 
	 * simply return it. True: copy GPU to CPU no matter what
	 */
	public float[] toHostArray(boolean forceCopy)
	{
		checkHostMode(HostMode.Array);
		
		if (hostArray != null && !forceCopy || device == null)
			return hostArray;
		if (hostArray == null) // create a new array
			hostArray = new float[size()];
		return GpuUtil.deviceToHostFloat(device, hostArray, size());
	}
	
	/**
	 * Copy to and return host array
	 * @see FloatMat#toHostArray(boolean) toHostArray(false)
	 */
	public float[] toHostArray() {	return toHostArray(false);	}
	
	/**
	 * Copy to and return host buffer
	 * Conflict exception if HostMode is Array
	 * If HostMode is None, set to Buffer
	 * If device is null, return current hostBuffer
	 * @param forceCopy default false: if hostBuffer already has a value, 
	 * simply return it. True: copy GPU to CPU no matter what
	 */
	public FloatBuffer toHostBuffer(boolean forceCopy)
	{
		checkHostMode(HostMode.Buffer);
		
		if (hostBuffer != null && !forceCopy || device == null)
			return hostBuffer;
		if (hostBuffer == null)
			hostBuffer = FloatBuffer.allocate(size());
		return GpuUtil.deviceToHostFloat(device, hostBuffer, size());
	}
	
	/**
	 * Copy to and return host buffer
	 * @see FloatMat#toHostBuffer(boolean) toHostBuffer(false)
	 */
	public FloatBuffer toHostBuffer() {	return toHostBuffer(false);	}
	
	/**
	 * Set the host array with HostMode check 
	 * you can disable by enableHostModeCheck(false)
	 */
	public void setHostArray(float[] hostArray)
	{
		checkHostMode(HostMode.Array);
		this.hostArray = hostArray;
	}
	
	/**
	 * @see FloatMat#setHostArray(float[])
	 */
	public void setHostArray(float[][] hostArray)
	{
		setHostArray(CpuUtil.flatten(hostArray, true));
	}
	
	/**
	 * Set the host buffer with HostMode check 
	 * you can disable by enableHostModeCheck(false)
	 */
	public void setHostBuffer(FloatBuffer hostBuffer)
	{
		checkHostMode(HostMode.Buffer);
		this.hostBuffer = hostBuffer;
	}
	
	/**
	 * Get a device pointer (wrapped in a FloatMat) 
	 * that starts from 'offset' and lasts 'size' floats.
	 * The shape might need to be adjusted. 
	 * Specify the number of rows, or leave it to be the current row dim.
	 * host, thrustPointer and transpose flag will be cleared.
	 * @param offMat output parameter
	 */
	public FloatMat createOffset(FloatMat offMat, int offset, int size, int newRow)
	{
		offMat.device = this.toDevice().withByteOffset(offset * Sizeof.FLOAT);
		offMat.hostMode = this.hostMode;
		offMat.initDim(newRow, size/newRow);
		return offMat;
	}
	
	/**
	 * Get a device pointer (wrapped in a FloatMat) 
	 * that starts from 'offset' and lasts 'size' floats.
	 * The shape might need to be adjusted. 
	 * Specify the number of rows, or leave it to be the current row dim.
	 * host, thrustPointer and transpose flag will be cleared.
	 */
	public FloatMat createOffset(int offset, int size, int newRow)
	{
		return createOffset(new FloatMat(), offset, size, newRow);
	}
	
	/**
	 * Default version of createOffset.
	 * Assume newRow to be the same as the current row dim. 
	 */
	public FloatMat createOffset(FloatMat offMat, int offset, int size)
	{
		return createOffset(offMat, offset, size, this.row);
	}
	
	/**
	 * Default version of createOffset.
	 * Assume newRow to be the same as the current row dim. 
	 */
	public FloatMat createOffset(int offset, int size)
	{
		return createOffset(offset, size, this.row);
	}
	
	/**
	 * createOffset from column 'start' to column 'end'
	 */
	public FloatMat createColOffset(FloatMat offMat, int colStart, int colEnd)
	{
		return createOffset(colStart * this.row, (colEnd - colStart) * this.row);
	}
	
	/**
	 * Default version of createColOffset
	 * @return new FloatMat
	 */
	public FloatMat createColOffset(int colStart, int colEnd)
	{
		return createColOffset(new FloatMat(), colStart, colEnd);
	}
	
	/**
	 * @return row * col
	 */
	public int size() { return row * col; }
	
	/**
	 * Free the device pointer
	 */
	public void destroy()
	{
		hostArray = null;
		hostBuffer = null;
		hostMode = HostMode.None;
		if (device != null)
		{
    		cudaFree(device);
    		device = null;
		}
		thrustPtr = null;
	}
	
	/**
	 * Static version of the instance method
	 * Destroy only when FloatMat isn't null
	 * @see FloatMat#destroy()
	 */
	public static void destroy(FloatMat mat)
	{
		if (mat != null && mat != DUMMY)
			mat.destroy();
	}
	
	/**
	 * Utility: flatten a 2D float array to 1D, column major
	 */
	public static float[] flatten(float[][] A)
	{
		return CpuUtil.flatten(A, true);
	}

	/**
	 * Utility: flatten a 2D float array to 1D, column major
	 * @param out output parameter
	 */
	public static float[] flatten(float[][] A, float[] out)
	{
		return CpuUtil.flatten(A, out, true);
	}
	
	/**
	 * Utility: deflatten a 1D float to 2D matrix, column major
	 */
	public static float[][] deflatten(float[] A, int row)
	{
		return CpuUtil.deflatten(A, row, true);
	}
	
	/**
	 * Utility: deflatten a 1D float to 2D matrix, column major
	 * @param out output parameter
	 */
	public static float[][] deflatten(float[] A, float[][] out)
	{
		return CpuUtil.deflatten(A, out, true);
	}
	
	/**
	 * Deflatten this to a 2D float array, column major
	 * Always force copy to hostArray
	 */
	public float[][] deflatten()
	{
		return deflatten(toHostArray(true), this.row);
	}
	
	/**
	 * @return its deflattened 2D float array representation
	 */
	public String toString()
	{
		return PP.mat2str(this.deflatten());
	}

	/**
	 * Transform an index to a coordinate (column major)
	 */
	public Coord toCoord(int idx)
	{
		return CpuUtil.toCoord(row, idx);
	}
	
	/**
	 * Transform a 2D coordinate to index (column major)
	 */
	public int toIndex(int i, int j)
	{
		return CpuUtil.toIndex(row, i, j);
	}
	/**
	 * Transform a 2D coordinate to index (column major)
	 */
	public int toIndex(Coord c)
	{
		return CpuUtil.toIndex(row, c);
	}
	
	// ******************** Interface to Thrust API ****************** /
	/**
	 * Get the thrust pointer
	 */
	public FloatDevicePointer getThrustPointer()
	{
		if (thrustPtr == null)
		{
			this.toDevice();
			thrustPtr = new FloatDevicePointer(this.device);
		}
		return thrustPtr;
	}
	
	/**
	 * exp(a * x + b)
	 */
	public FloatMat exp(float a, float b)
	{
		Thrust.exp(this, a, b); return this;
	}
	public FloatMat exp()
	{
		Thrust.exp(this); return this;
	}
	
	/**
	 * log(a * x + b)
	 */
	public FloatMat log(float a, float b)
	{
		Thrust.log(this, a, b); return this;
	}
	public FloatMat log()
	{
		Thrust.log(this); return this;
	}
	
	/**
	 * cos(a * x + b)
	 */
	public FloatMat cos(float a, float b)
	{
		Thrust.cos(this, a, b); return this;
	}
	public FloatMat cos()
	{
		Thrust.cos(this); return this;
	}
	
	/**
	 * sin(a * x + b)
	 */
	public FloatMat sin(float a, float b)
	{
		Thrust.sin(this, a, b); return this;
	}
	public FloatMat sin()
	{
		Thrust.sin(this); return this;
	}
	
	/**
	 * abs(a * x + b)
	 */
	public FloatMat abs(float a, float b)
	{
		Thrust.abs(this, a, b); return this;
	}
	public FloatMat abs()
	{
		Thrust.abs(this); return this;
	}
	
	/**
	 * sqrt(a * x + b)
	 */
	public FloatMat sqrt(float a, float b)
	{
		Thrust.sqrt(this, a, b); return this;
	}
	public FloatMat sqrt()
	{
		Thrust.sqrt(this); return this;
	}
	
	/**
	 * (a * x + b)^2
	 */
	public FloatMat square(float a, float b)
	{
		Thrust.square(this, a, b); return this;
	}
	public FloatMat square()
	{
		Thrust.square(this); return this;
	}
	
	/**
	 * (a * x + b)^3
	 */
	public FloatMat cube(float a, float b)
	{
		Thrust.cube(this, a, b); return this;
	}
	public FloatMat cube()
	{
		Thrust.cube(this); return this;
	}
	
	/**
	 * 1 / (a * x + b)
	 */
	public FloatMat reciprocal(float a, float b)
	{
		Thrust.reciprocal(this, a, b); return this;
	}
	public FloatMat reciprocal()
	{
		Thrust.reciprocal(this); return this;
	}
	
	/**
	 * (a * x + b) ^p
	 */
	public FloatMat pow(float p, float a, float b)
	{
		Thrust.pow(this, p, a, b); return this;
	}
	public FloatMat pow(float p)
	{
		Thrust.pow(this, p); return this;
	}
	
	/**
	 * (a * x + b)
	 */
	public FloatMat linear(float a, float b)
	{
		Thrust.linear(this, a, b); return this;
	}
	
	/**
	 * Sigmoid(a * x + b)
	 */
	public FloatMat sigmoid(float a, float b)
	{
		Thrust.sigmoid(this, a, b); return this;
	}
	public FloatMat sigmoid()
	{
		Thrust.sigmoid(this); return this;
	}
	
	/**
	 * x .* (1 - x)
	 */
	public FloatMat sigmoid_deriv(float a, float b)
	{
		Thrust.sigmoid_deriv(this, a, b); return this;
	}
	public FloatMat sigmoid_deriv()
	{
		Thrust.sigmoid_deriv(this); return this;
	}
	
	public float max()
	{
		return Thrust.max(this);
	}

	public float min()
	{
		return Thrust.min(this);
	}

	public float sum()
	{
		return Thrust.sum(this);
	}

	public float product()
	{
		return Thrust.product(this);
	}
	
	public FloatMat sort()
	{
		Thrust.sort(this);	return this;
	}
	
	public FloatMat fill(float val)
	{
		Thrust.fill(this, val);	return this;
	}
	
	public FloatMat copyFrom(FloatMat other)
	{
		Thrust.copy(other, this);	return this;
	}
	
	/**
	 * Set a single value to newVal
	 */
	public FloatMat singleSet(int i, int j, float newVal)
	{
		Thrust.single_set(this, i, j, newVal); return this;
	}
	/**
	 * Set a single value to newVal
	 */
	public FloatMat singleSet(int idx, float newVal)
	{
		Thrust.single_set(this, idx, newVal); return this;
	}
	
	/**
	 * Increment a single value to newVal
	 */
	public FloatMat singleIncr(int i, int j, float newVal)
	{
		Thrust.single_incr(this, i, j, newVal); return this;
	}
	/**
	 * Increment a single value to newVal
	 */
	public FloatMat singleIncr(int idx, float newVal)
	{
		Thrust.single_incr(this, idx, newVal); return this;
	}
}
