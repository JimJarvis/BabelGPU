package gpu;

import gpu.ThrustStruct.DoubleDevicePointer;
import utils.CpuUtil;
import utils.CpuUtil.Coord;
import utils.GpuUtil;
import utils.PP;
import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.runtime.JCuda.*;
import static jcuda.jcublas.cublasOperation.*;

/**
 * Struct around a matrix with row/col dimension info
 */
public class DoubleMat
{
	private double[] host = null;
	private Pointer device = null; // jCuda pointer
	private DoubleDevicePointer thrustPointer = null; // Thrust pointer
	
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
	public DoubleMat() {	}
	
	/**
	 * Ctor from host data
	 */
	public DoubleMat(double[] host, int row, int col)
	{
		this.host = host;
		initDim(row, col);
	}
	
	/**
	 * Ctor from 2D host data
	 */
	public DoubleMat(double[][] host)
	{
		this(flatten(host), host.length, host[0].length);
	}
	
	/**
	 * Ctor for 1D vector (column vector)
	 */
	public DoubleMat(double[] host)
	{
		this(host, host.length, 1);
	}

	/**
	 * Ctor from device data
	 */
	public DoubleMat(Pointer device, int row, int col)
	{
		this.device = device;
		initDim(row, col);
	}
	
	/**
	 * Ctor from device data: 1D vector (column vector)
	 */
	public DoubleMat(Pointer device, int len)
	{
		this(device, len, 1);
	}
	
	/**
	 * Ctor with dimensions
	 * @param memsetToZero true to initialize the device data to 0. Default true. 
	 */
	public DoubleMat(int row, int col, boolean memsetToZero)
	{
		this.device = GpuUtil.allocDeviceDouble(row * col, memsetToZero);
		initDim(row, col);
	}
	
	/**
	 * Ctor with dimensions
	 * The device data will be initialized to all 0
	 */
	public DoubleMat(int row, int col)
	{
		this(row, col, true);
	}
	
	/**
	 * Instantiate a new empty DoubleMat with the same size
	 * NOTE: doesn't copy any data. Only the same row/col
	 */
	public DoubleMat(DoubleMat other)
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
	private DoubleMat shallowCopy()
	{
		DoubleMat mat = new DoubleMat();
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
	public DoubleMat transpose()
	{
		// Swap row and col dimension
		DoubleMat mat = this.shallowCopy();
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
		if (device != null)
			GpuUtil.clearDeviceDouble(device, size());
	}
	
	/**
	 * Get the device pointer
	 * If 'device' field is currently null, we copy host to GPU
	 */
	public Pointer getDevice()
	{
		if (device == null)
			device = GpuBlas.hostToCublasDouble(host);
		return device;
	}
	
	/**
	 * Get the host pointer
	 * If host is currently null, we copy device to CPU
	 */
	public double[] getHost()
	{
		if (host == null)
			host = GpuBlas.cublasToHostDouble(device, size());
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
			GpuBlas.hostToCublasDouble(host, device);
		}
		else // device is null
    		device = GpuBlas.hostToCublasDouble(host);
		return device;
	}
	
	/**
	 * No matter whether 'host' field is null or not, we copy device to CPU
	 * Syncs host w.r.t. device
	 * @return host array
	 */
	public double[] copyDeviceToHost()
	{
		if (device == null) 	return null;
		if (host != null)
			GpuBlas.cublasToHostDouble(device, host);
		else // host is null
			host = GpuBlas.cublasToHostDouble(device, size());
		
		return host;
	}
	
	/**
	 * Get a device pointer (wrapped in a DoubleMat) 
	 * that starts from 'offset' and lasts 'size' doubles.
	 * The shape might need to be adjusted. 
	 * Specify the number of rows, or leave it to be the current row dim.
	 * host, thrustPointer and transpose flag will be cleared.
	 * @param offMat output parameter
	 */
	public DoubleMat createOffset(DoubleMat offMat, int offset, int size, int newRow)
	{
		offMat.device = this.getDevice().withByteOffset(offset * Sizeof.DOUBLE);
		offMat.initDim(newRow, size/newRow);
		return offMat;
	}
	
	/**
	 * Get a device pointer (wrapped in a DoubleMat) 
	 * that starts from 'offset' and lasts 'size' doubles.
	 * The shape might need to be adjusted. 
	 * Specify the number of rows, or leave it to be the current row dim.
	 * host, thrustPointer and transpose flag will be cleared.
	 */
	public DoubleMat createOffset(int offset, int size, int newRow)
	{
		return createOffset(new DoubleMat(), offset, size, newRow);
	}
	
	/**
	 * Default version of createOffset.
	 * Assume newRow to be the same as the current row dim. 
	 */
	public DoubleMat createOffset(DoubleMat offMat, int offset, int size)
	{
		return createOffset(offMat, offset, size, this.row);
	}
	
	/**
	 * Default version of createOffset.
	 * Assume newRow to be the same as the current row dim. 
	 */
	public DoubleMat createOffset(int offset, int size)
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
	 * Utility: flatten a 2D double array to 1D, column major
	 */
	public static double[] flatten(double[][] A)
	{
		return CpuUtil.flatten(A, true);
	}

	/**
	 * Utility: flatten a 2D double array to 1D, column major
	 * @param out output parameter
	 */
	public static double[] flatten(double[][] A, double[] out)
	{
		return CpuUtil.flatten(A, out, true);
	}
	
	/**
	 * Utility: deflatten a 1D double to 2D matrix, column major
	 */
	public static double[][] deflatten(double[] A, int row)
	{
		return CpuUtil.deflatten(A, row, true);
	}
	
	/**
	 * Utility: deflatten a 1D double to 2D matrix, column major
	 * @param out output parameter
	 */
	public static double[][] deflatten(double[] A, double[][] out)
	{
		return CpuUtil.deflatten(A, out, true);
	}
	
	/**
	 * Deflatten this to a 2D double array, column major
	 */
	public double[][] deflatten()
	{
		return deflatten(device == null ? 
				getHost() : copyDeviceToHost(), this.row);
	}
	
	/**
	 * @return its deflattened 2D double array representation
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
	public DoubleDevicePointer getThrustPointer()
	{
		if (thrustPointer == null)
		{
			if (device == null) // initialize device
				this.getDevice();
			thrustPointer = new DoubleDevicePointer(this.device);
		}
		return thrustPointer;
	}
	
	/**
	 * exp(a * x + b)
	 */
	public DoubleMat exp(double a, double b)
	{
		Thrust.exp(this, a, b); return this;
	}
	public DoubleMat exp()
	{
		Thrust.exp(this); return this;
	}
	
	/**
	 * log(a * x + b)
	 */
	public DoubleMat log(double a, double b)
	{
		Thrust.log(this, a, b); return this;
	}
	public DoubleMat log()
	{
		Thrust.log(this); return this;
	}
	
	/**
	 * cos(a * x + b)
	 */
	public DoubleMat cos(double a, double b)
	{
		Thrust.cos(this, a, b); return this;
	}
	public DoubleMat cos()
	{
		Thrust.cos(this); return this;
	}
	
	/**
	 * sin(a * x + b)
	 */
	public DoubleMat sin(double a, double b)
	{
		Thrust.sin(this, a, b); return this;
	}
	public DoubleMat sin()
	{
		Thrust.sin(this); return this;
	}
	
	/**
	 * sqrt(a * x + b)
	 */
	public DoubleMat sqrt(double a, double b)
	{
		Thrust.sqrt(this, a, b); return this;
	}
	public DoubleMat sqrt()
	{
		Thrust.sqrt(this); return this;
	}
	
	/**
	 * abs(a * x + b)
	 */
	public DoubleMat abs(double a, double b)
	{
		Thrust.abs(this, a, b); return this;
	}
	public DoubleMat abs()
	{
		Thrust.abs(this); return this;
	}
	
	/**
	 * (a * x + b) ^p
	 */
	public DoubleMat pow(double p, double a, double b)
	{
		Thrust.pow(this, p, a, b); return this;
	}
	public DoubleMat pow(double p)
	{
		Thrust.pow(this, p); return this;
	}
	
	/**
	 * (a * x + b)
	 */
	public DoubleMat linear(double a, double b)
	{
		Thrust.linear(this, a, b); return this;
	}
	
	public double max()
	{
		return Thrust.max(this);
	}

	public double min()
	{
		return Thrust.min(this);
	}

	public double sum()
	{
		return Thrust.sum(this);
	}

	public double product()
	{
		return Thrust.product(this);
	}
	
	public DoubleMat sort()
	{
		Thrust.sort(this);	return this;
	}
	
	public DoubleMat fill(double val)
	{
		Thrust.fill(this, val);	return this;
	}
	
	public DoubleMat copyFrom(DoubleMat other)
	{
		Thrust.copy(other, this);	return this;
	}
}
