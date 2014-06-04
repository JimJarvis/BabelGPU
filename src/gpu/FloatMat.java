package gpu;

import gpu.ThrustStruct.FloatDevicePointer;
import utils.CpuUtil.Coord;
import utils.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.runtime.JCuda.*;
import static jcuda.jcublas.cublasOperation.*;

/**
 * Struct around a matrix with row/col dimension info
 */
public class FloatMat
{
	private float[] host = null;
	private Pointer device = null; // jCuda pointer
	private FloatDevicePointer thrustPointer = null; // Thrust pointer
	
	// This field records whether the matrix should be transposed or not
	private int op = CUBLAS_OP_N; 
	public int row;
	public int col;
	// Leading dimension: column length (i.e. row dim)
	// Doesn't change even with transpose
	public int ldim; 
	
	/**
	 * Default ctor
	 */
	public FloatMat() {	}
	
	/**
	 * Ctor from host data
	 */
	public FloatMat(float[] host, int row, int col)
	{
		this.host = host;
		initDim(row, col);
	}
	
	/**
	 * Ctor from 2D host data
	 */
	public FloatMat(float[][] host)
	{
		this(flatten(host), host.length, host[0].length);
	}
	
	/**
	 * Ctor for 1D vector (column vector)
	 */
	public FloatMat(float[] host)
	{
		this(host, host.length, 1);
	}

	/**
	 * Ctor from device data
	 */
	public FloatMat(Pointer device, int row, int col)
	{
		this.device = device;
		initDim(row, col);
	}
	
	/**
	 * Ctor from device data: 1D vector (column vector)
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
		mat.host = this.host;
		mat.thrustPointer = this.thrustPointer;
		
		return mat;
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
	 * Get the device pointer
	 * If 'device' field is currently null, we copy host to GPU
	 */
	public Pointer getDevice()
	{
		if (device == null)
			device = GpuBlas.hostToCublasFloat(host);
		return device;
	}

	/**
	 * Get the host pointer
	 * If host is currently null, we copy device to CPU
	 */
	public float[] getHost()
	{
		if (host == null)
			host = GpuBlas.cublasToHostFloat(device, size());
		return host;
	}
		
	/**
	 * No matter whether 'device' field is null or not, we copy host to GPU
	 * Syncs device w.r.t. host
	 * @return device pointer
	 */
	public Pointer copyHostToDevice()
	{
		if (host == null)  return null;
		if (device != null)
		{
			cudaFree(device);
			GpuBlas.hostToCublasFloat(host, device);
		}
		else // device is null
    		device = GpuBlas.hostToCublasFloat(host);
		return device;
	}
	
	/**
	 * No matter whether 'host' field is null or not, we copy device to CPU
	 * Syncs host w.r.t. device
	 * @return host array
	 */
	public float[] copyDeviceToHost()
	{
		if (device == null) 	return null;
		if (host != null)
			GpuBlas.cublasToHostFloat(device, host);
		else // host is null
			host = GpuBlas.cublasToHostFloat(device, size());
		
		return host;
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
		offMat.device = this.getDevice().withByteOffset(offset * Sizeof.FLOAT);
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
	 * @return row * col
	 */
	public int size() { return row * col; }
	
	/**
	 * Free the device pointer
	 */
	public void destroy()
	{
		host = null;
		if (device != null)
		{
    		cudaFree(device);
    		device = null;
		}
		thrustPointer = null;
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
	 */
	public float[][] deflatten()
	{
		return deflatten(device == null ? 
				getHost() : copyDeviceToHost(), this.row);
	}
	
	/**
	 * @return its deflattened 2D float array representation
	 */
	public String toString()
	{
		return PP.o2str(this.deflatten());
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
		if (thrustPointer == null)
		{
			if (device == null) // initialize device
				this.getDevice();
			thrustPointer = new FloatDevicePointer(this.device);
		}
		return thrustPointer;
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
}
