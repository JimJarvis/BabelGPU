package gpu;

import gpu.ThrustStruct.FloatDevicePointer;
import utils.GpuUtil;
import utils.PP;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.jcublas.cublasOperation;

/**
 * Struct around a matrix with row/col dimension info
 */
public class FloatMat
{
	private Pointer hostPtr = null; // jCuda pointer
	private Pointer devicePtr = null; // jCuda pointer
	private FloatDevicePointer thrustPointer = null; // Thrust pointer
	
	// This field records whether the matrix should be transposed or not
	private int op = cublasOperation.CUBLAS_OP_N; 
	public int numRows;
	public int numCols;
	// Leading dimension: column length (i.e. row dim)
	// Doesn't change even with transpose
	public int ldim; 
	
	
	// used internally for shallow copies.
	private FloatMat() { }

	/**
	 * Ctor with dimensions
	 * This constructor doesn't set the hostPtr.
	 * Use this constructor in the scenario where you won't need to read the 
	 * data back from the GPU (or, if you do need to, you can just provide a 
	 * float[] hostArray to the copyDeviceToHostArray method)
	 * @param memsetToZero true to initialize the device data to 0. Default false. 
	 * @throws GpuException 
	 */
	public FloatMat(int row, int col, boolean memsetToZero) throws GpuException
	{
		this.devicePtr = GpuUtil.allocateDeviceFloat(row * col, memsetToZero);
		initDim(row, col);
	}
	
	public FloatMat(Pointer hostPtr, Pointer devicePtr, int row, int col) throws GpuException
	{
		this.hostPtr = hostPtr;
		this.devicePtr = devicePtr;
		initDim(row, col);
	}
	
	/**
	 * Ctor from host data
	 * This constructor assumes the memory on the device has already been
	 * allocated, and that the 'device' Pointer to that memory is passed in. 
	 * @throws GpuException 
	 */
	public FloatMat(Pointer hostPtr, int row, int col, boolean memsetToZero) throws GpuException
	{
		this.hostPtr = hostPtr;
		this.devicePtr = GpuUtil.allocateDeviceFloat(row * col, memsetToZero);
		initDim(row, col);
	}
	
	/**
	 * Ctor from host data
	 * @param memsetToZero true to initialize the device data to 0. Default false. 
	 * @throws GpuException 
	 */
	public FloatMat(float[] host, int row, int col, boolean memsetToZero) throws GpuException
	{
		this.hostPtr = Pointer.to(host);
		this.devicePtr = GpuUtil.allocateDeviceFloat(row * col, memsetToZero);
		initDim(row, col);
	}
	
	/**
	 * Ctor from host data
	 * @throws GpuException 
	 */
	public FloatMat(float[] host, int row, int col) throws GpuException
	{
		this(host, row, col, false /*memSetToZero*/);
	}
	
	/**
	 * Ctor from 2D host data
	 * Note: the flatten method is very inefficient, and shouldn't be used.
	 * @throws GpuException 
	 */
	public FloatMat(float[][] host) throws GpuException
	{
		this(flatten(host), host.length, host[0].length);
	}
	
	/**
	 * Ctor for 1D host data (column vector)
	 */
	public FloatMat(float[] host) throws GpuException
	{
		this(host, host.length, 1);
	}

	// Ctor helper
	private void initDim(int row, int col)
	{
		this.numRows = row;
		this.numCols = col;
		this.ldim = row;
	}
	
	// Shallow copy create new instance
	private FloatMat shallowCopy()
	{
		FloatMat mat = new FloatMat();
		mat.numRows = this.numRows;
		mat.numCols = this.numCols;
		mat.ldim = this.ldim;
		mat.op = this.op;
		mat.devicePtr = this.devicePtr;
		mat.hostPtr = this.hostPtr;
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
		mat.numRows = this.numCols;
		mat.numCols = this.numRows;
		mat.op = (this.op != cublasOperation.CUBLAS_OP_N) ? 
				cublasOperation.CUBLAS_OP_N : cublasOperation.CUBLAS_OP_T;
		return mat;
	}
	
	public boolean isTransposed()
	{
		return this.op == cublasOperation.CUBLAS_OP_T;
	}
	
	
	public int getOp() 
	{
		return this.op;	
	}
	
	/**
	 * Invariant to transpose
	 */
	public int getOriginalRow()
	{
		return this.op == cublasOperation.CUBLAS_OP_N ? this.numRows : this.numCols;
	}

	/**
	 * Invariant to transpose
	 */
	public int getOriginalCol()
	{
		return this.op == cublasOperation.CUBLAS_OP_N ? this.numCols : this.numRows;
	}
	
	public void copyHostToDevice() throws GpuException
	{
		this.copyHostToDevice(this.size());
	}
	
	public void copyHostToDevice(int numFloatsToCopy) throws GpuException
	{
		if(this.hostPtr == null)
		{
			throw new GpuException("Cannot copy host to device if hostPtr is null");
		}
		
		GpuUtil.hostToDeviceFloat(
				this.devicePtr, 
				this.hostPtr, 
				numFloatsToCopy
			);
	}
	
	public void copyDeviceToHostArray(/*ref*/ float[] hostArray) throws GpuException
	{
		GpuUtil.deviceToHostFloat(this.devicePtr, /*ref*/ hostArray);
	}
	
	public void copyDeviceToHost(int numFloatsToCopy) throws GpuException
	{
		if(this.hostPtr == null)
		{
			throw new GpuException("Cannot copy device to host if hostPtr is null");
		}
		
		GpuUtil.deviceToHostFloat(this.devicePtr, this.hostPtr, numFloatsToCopy);
	}
	
	public Pointer getDevicePointer()
	{
		return this.devicePtr;
	}
	
	public Pointer getHostPointer()
	{
		return this.hostPtr;
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
		FloatMat off = new FloatMat();
		off.devicePtr = this.getDevicePointer().withByteOffset(offset * Sizeof.FLOAT);
		off.initDim(newRow, size/newRow);
		return off;
	}
	
	/**
	 * Default version of createOffset.
	 * Assume newRow to be the same as the current row dim. 
	 */
	public FloatMat createOffset(int offset, int size)
	{
		return createOffset(offset, size, this.numRows);
	}
	
	/**
	 * @return row * col
	 */
	public int size() 
	{ 
		return this.numRows * this.numCols; 
	}
	
	/**
	 * Free the device pointer
	 */
	public void destroy()
	{
		this.hostPtr = null;
		JCuda.cudaFree(this.devicePtr);
		this.devicePtr = null;
		this.thrustPointer = null;
	}
	
	/**
	 * Utility: flatten a 2D float array to 1D, column major
	 */
	public static float[] flatten(float[][] A)
	{
		PP.out.println("The flatten method is super inefficient.  Don't use this in production code");
		
		int row = A.length;
		int col = A[0].length;
		float[] ans = new float[row * col];
		int pt = 0;

		for (int j = 0; j < col; j ++)
			for (int i = 0; i < row; i ++)
				ans[pt ++] = A[i][j];

		return ans;
	}
	
	/**
	 * Utility: deflatten a 1D float to 2D matrix, column major
	 */
	public static void deflatten(
			float[] A, 
			/*ref*/ float[][] deflattenedHostArray
		)
	{
		int numRows = deflattenedHostArray.length;
		int numCols = deflattenedHostArray[0].length;
		int pt = 0;
		
		for (int j = 0; j < numCols; j ++)
		{
			for (int i = 0; i < numRows; i ++)
			{
				deflattenedHostArray[i][j] = A[pt ++];
			}
		}
	}
	
	/**
	 * Deflatten this to a 2D float array, column major
	 * @throws GpuException 
	 */
	public void copyDeviceToHostArrayAndDeflatten(
			/*ref*/ float[] hostArray,
			/*ref*/ float[][] deflattenedHostArray
			) throws GpuException
	{
		this.copyDeviceToHostArray(/*ref*/ hostArray);
		deflatten(hostArray, deflattenedHostArray);
	}
	
	/**
	 * @return its deflattened 2D float array representation
	 */
	public String toString()
	{
		try 
		{
			float[][] deflattened = new float[this.numRows][this.numCols];
			this.copyDeviceToHostArrayAndDeflatten(
					/*ref*/ new float[this.size()], 
					/*ref*/ deflattened
				);
			
			return PP.o2str(deflattened);
		} 
		catch (GpuException e) 
		{
			e.printStackTrace();
		}
		return null;
	}

	
	/**
	 * Inner class for 2D coordinate in the matrix
	 */
	public static class Coord
	{
		public int i; // row
		public int j; // col
		public Coord(int i, int j)
		{
			this.i = i; 
			this.j = j;
		}
		
		public String toString() { return String.format("<%d, %d>", i, j); }
	}
	
	/**
	 * Transform an index to a coordinate (column major)
	 */
	public Coord toCoord(int idx)
	{
		return new Coord(idx % this.numRows, idx / this.numRows);
	}
	
	/**
	 * Transform a 2D coordinate to index (column major)
	 */
	public int toIndex(int i, int j)
	{
		return j * this.numRows + i;
	}
	/**
	 * Transform a 2D coordinate to index (column major)
	 */
	public int toIndex(Coord c)
	{
		return c.j * this.numRows + c.i;
	}
	
	// ******************** Interface to Thrust API ****************** /
	/**
	 * Get the thrust pointer
	 * Note: the host pointer must be set before calling this.
	 * @throws GpuException 
	 */
	public FloatDevicePointer getThrustPointer() throws GpuException
	{
		if (this.thrustPointer == null)
		{
			if (this.devicePtr == null)
			{
				this.copyHostToDevice(this.size());
			}
			this.thrustPointer = new FloatDevicePointer(this.devicePtr);
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
