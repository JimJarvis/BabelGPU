package gpu;

import java.nio.FloatBuffer;

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
	// on the host side, the data can be stored as either a float[] or a FloatBuffer 
	private float[] hostArray = null;
	private FloatBuffer hostBuffer = null;
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
	
	
	// used for shallow copies.
	public FloatMat() { }

	/**
	 * Ctor from host and device data
	 * In this constructor, both the hostPtr and the devicePtr are passed in.
	 * The assumption is that memory has been allocated on the host and on the device already.
	 */
	public FloatMat(
			Pointer devicePtr, Pointer hostPtr,  
			float[] hostArray, FloatBuffer hostBuffer,
			int numRows, int numCols) throws GpuException
	{
		this.devicePtr = devicePtr;
		
		if(hostPtr == null)
		{
			if(hostBuffer != null)
			{
				this.hostPtr = Pointer.toBuffer(hostBuffer);
			}
			else if(hostArray != null)
			{
				this.hostPtr = Pointer.to(hostArray);
			}
		}
		else
		{
			this.hostPtr = hostPtr;
		}
		
		this.hostArray = hostArray;
		this.hostBuffer = hostBuffer;
		this.initDim(numRows, numCols);
	}
	
	/**
	 * Ctor from host data
	 * This constructor assumes the memory on the host has already been
	 * allocated, and that the 'hostPtr' pointer to that memory is passed in. 
	 * It allocates numRows * numCols * sizeOf.FLOAT memory on the device.
	 * @param memsetToZero true to initialize the device data to 0.
	 * @throws GpuException 
	 */
	public FloatMat(Pointer hostPtr, int numRows, int numCols, boolean memsetToZero) throws GpuException
	{
		this(
				GpuUtil.allocateDeviceFloat(numRows * numCols, memsetToZero), 
				hostPtr, 
				null, // hostArray
				null, // hostBuffer
				numRows, 
				numCols
			);
	}
	
	/**
	 * Ctor from host data
	 * This constructor assumes the memory on the host has already been
	 * allocated as a FloatBuffer (which is passed in as parameter).
	 * It allocates numRows * numCols * sizeOf.FLOAT memory on the device. 
	 * @param memsetToZero true to initialize the device data to 0.
	 * @throws GpuException 
	 */
	public FloatMat(FloatBuffer hostBuffer, int numRows, int numCols, boolean memsetToZero) throws GpuException
	{
		// TODO: should we use "Pointer.to" or "Pointer.toBuffer"
		this(
				GpuUtil.allocateDeviceFloat(numRows * numCols, memsetToZero), 
				Pointer.toBuffer(hostBuffer), 
				null, // hostArray
				hostBuffer, // hostBuffer
				numRows, 
				numCols
			);
	}
	
	/**
	 * Ctor from host data
	 * This constructor assumes the memory on the host has already been
	 * allocated as a 1D array (which is passed in as parameter).
	 * It allocates numRows * numCols * sizeOf.FLOAT memory on the device. 
	 * @param memsetToZero true to initialize the device data to 0.
	 * @throws GpuException 
	 */
	public FloatMat(float[] hostArray, int numRows, int numCols, boolean memsetToZero) throws GpuException
	{
		this(
				GpuUtil.allocateDeviceFloat(numRows * numCols, memsetToZero), 
				Pointer.to(hostArray), 
				hostArray, // hostArray
				null, // hostBuffer
				numRows, 
				numCols
			);
	}
	
	/**
	 * Ctor from 2D host data
	 * This constructor assumes the memory on the host has already been
	 * allocated as a 2D array (which is passed in as parameter).
	 * It allocates numRows * numCols * sizeOf.FLOAT memory on the device. 
	 * Note: the flatten method is very inefficient, and shouldn't be used.
	 * @throws GpuException 
	 */
	public FloatMat(float[][] hostArray) throws GpuException
	{
		this(
				GpuUtil.allocateDeviceFloat(hostArray.length * hostArray[0].length, false /*memsetToZero*/),
				null, // hostPtr (will be set to point to flatten(hostArray) in the constructor
				flatten(hostArray), // hostArray
				null, // hostBuffer
				hostArray.length, 
				hostArray[0].length
			);
	}
	
	/**
	 * Ctor for 1D host data (column vector)
	 */
	public FloatMat(float[] hostArray) throws GpuException
	{
		this(
				GpuUtil.allocateDeviceFloat(hostArray.length, false /*memsetToZero*/), 
				Pointer.to(hostArray), 
				hostArray, // hostArray
				null, // hostBuffer
				hostArray.length, 
				1 // numCols
			);
	}
	
	/**
	 * Ctor with dimensions
	 * This constructor doesn't set the hostPtr.
	 * Use this constructor in the scenario where you won't need to read the 
	 * data back from the GPU (or, if you do need to, you can just provide a 
	 * float[] hostArray to the copyDeviceToHostArray method)
	 * @param memsetToZero true to initialize the device data to 0. Default false. 
	 * @throws GpuException 
	 */
	public FloatMat(int numRows, int numCols, boolean memsetToZero) throws GpuException
	{
		this(
				GpuUtil.allocateDeviceFloat(numRows * numCols, memsetToZero),
				null, //hostPtr
				null, // hostArray
				null, // hostBuffer
				numRows,
				numCols
			);
	}
	
	// Shallow copy create new instance
	private FloatMat shallowCopy()
	{
		FloatMat mat = new FloatMat();
		mat.hostPtr = this.hostPtr;
		mat.devicePtr = this.devicePtr;
		mat.hostArray = this.hostArray;
		mat.hostBuffer = this.hostBuffer;
		mat.thrustPointer = this.thrustPointer;
		mat.numRows = this.numRows;
		mat.numCols = this.numCols;
		mat.ldim = this.ldim;
		mat.op = this.op;
		
		return mat;
	}
	
	// Ctor helper
	private void initDim(int row, int col)
	{
		this.numRows = row;
		this.numCols = col;
		this.ldim = row;
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
	
	public Pointer getDevicePointer()
	{
		return this.devicePtr;
	}
	
	public Pointer getHostPointer()
	{
		return this.hostPtr;
	}
	
	public int getOp()
	{
		return this.op;
	}
	
	public boolean isTransposed()
	{
		return this.op == cublasOperation.CUBLAS_OP_T;
	}
	
	/**
	 * Invariant to transpose
	 */
	public int getOriginalNumRows()
	{
		return (this.op == cublasOperation.CUBLAS_OP_N) ? this.numRows : this.numCols;
	}

	/**
	 * Invariant to transpose
	 */
	public int getOriginalNumCols()
	{
		return (this.op == cublasOperation.CUBLAS_OP_N) ? this.numCols : this.numRows;
	}
	
	public void setHostArray(float[] hostArray)
	{
		this.hostPtr = Pointer.to(hostArray);
	}
	
	/**
	 * Set the memory of the device pointer to 0
	 */
	public void clearDevice()
	{
		if (devicePtr != null)
			GpuUtil.clearDeviceFloat(devicePtr, size());
	}
	
	// HOST -> GPU
	public void copyHostToDevice() throws GpuException
	{
		this.copyHostToDevice(this.size());
	}
	
	// HOST -> GPU
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
	
	// GPU -> HOST
	public float[] copyDeviceToHostAndReturnHostArray() throws GpuException
	{
		if(this.hostArray == null)
		{
			throw new GpuException("this.hostArray is null");
		}
		
		this.copyDeviceToHost();
		return this.hostArray;
	}

	// GPU -> HOST
	public FloatBuffer copyDeviceToHostAndReturnHostBuffer() throws GpuException
	{
		if(this.hostBuffer == null)
		{
			throw new GpuException("this.hostBuffer is null");
		}
		
		this.copyDeviceToHost();
		return this.hostBuffer;
	}
	
	// GPU -> HOST
	public void copyDeviceToHost() throws GpuException
	{
		this.copyDeviceToHost(this.size());
	}
	
	// GPU -> HOST
	public void copyDeviceToHost(int numFloatsToCopy) throws GpuException
	{
		if(this.hostPtr == null)
		{
			throw new GpuException("Cannot copy device to host if hostPtr is null");
		}
		
		GpuUtil.deviceToHostFloat(this.devicePtr, this.hostPtr, numFloatsToCopy);
	}
	
	// GPU -> HOST
	public void copyDeviceToHostArray(/*ref*/ float[] hostArray) throws GpuException
	{		
		GpuUtil.deviceToHostFloat(this.devicePtr, Pointer.to(hostArray), hostArray.length);
	}
	
	public float[] getHostArray()
	{
		return this.hostArray;
	}
	
	public FloatBuffer getHostBuffer()
	{
		return this.hostBuffer;
	}
	
	/**
	 * Get a device pointer (wrapped in a FloatMat) 
	 * that starts from 'offset' and lasts 'size' floats.
	 * The shape might need to be adjusted. 
	 * Specify the number of rows, or leave it to be the current row dim.
	 * host, thrustPointer and transpose flag will be cleared.
	 */
	public void createOffset(/*ref*/ FloatMat offsetPtr, int offset, int size, int newNumRows)
	{
		offsetPtr.devicePtr = this.devicePtr.withByteOffset(offset * Sizeof.FLOAT);
		offsetPtr.initDim(newNumRows, size/newNumRows);
	}
	
	/**
	 * Get a device pointer (wrapped in a FloatMat) 
	 * that starts from 'offset' and lasts 'size' floats.
	 * The shape might need to be adjusted. 
	 * Specify the number of rows, or leave it to be the current row dim.
	 * host, thrustPointer and transpose flag will be cleared.
	 */
	public FloatMat createOffset(int offset, int size, int newNumRows)
	{
		FloatMat offsetPtr = new FloatMat();
		this.createOffset(offsetPtr, offset, size, newNumRows);
		return offsetPtr;
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
	 * @return numRows * numCols
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
		
		int numRows = A.length;
		int numCols = A[0].length;
		float[] ans = new float[numRows * numCols];
		int pt = 0;

		for (int j = 0; j < numCols; j ++)
			for (int i = 0; i < numRows; i ++)
				ans[pt ++] = A[i][j];

		return ans;
	}
	
	/**
	 * Utility: deflatten the 1D hostArray into a 2D array of the appropriate dimensions.
	 * hostArray is assumed to be in column major format.
	 */
	public float[][] deflatten() throws BabelGpuException
	{
		if(this.hostArray == null && this.hostBuffer == null)
		{
			throw new BabelGpuException("Can't deflatten FloatMat with no host data.  Please copy data to CPU before calling this");
		}
		
		if(this.hostArray == null && this.hostBuffer != null)
		{
			this.hostArray = new float[this.size()];
			this.hostBuffer.clear();
			this.hostBuffer.get(this.hostArray);
		}
		
		float[][] deflattenedHostArray = new float[this.numRows][this.numCols];
		FloatMat.deflatten(this.hostArray, /*ref*/ deflattenedHostArray);
		return deflattenedHostArray;
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
	 * Utility: deflatten a 1D float to 2D matrix, column major
	 */
	public static float[][] deflatten(float[] A, int numRows)
	{
		float[][] deflattenedHostArray = new float[numRows][A.length/numRows];
		FloatMat.deflatten(A, /*ref*/ deflattenedHostArray);
		return deflattenedHostArray;
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
		public int i; // numRows
		public int j; // numCols
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
	public FloatMat exp(float a, float b) throws GpuException
	{
		Thrust.exp(this, a, b); return this;
	}
	public FloatMat exp() throws GpuException
	{
		Thrust.exp(this); return this;
	}
	
	/**
	 * log(a * x + b)
	 */
	public FloatMat log(float a, float b) throws GpuException
	{
		Thrust.log(this, a, b); return this;
	}
	public FloatMat log() throws GpuException
	{
		Thrust.log(this); return this;
	}
	
	/**
	 * cos(a * x + b)
	 */
	public FloatMat cos(float a, float b) throws GpuException
	{
		Thrust.cos(this, a, b); return this;
	}
	public FloatMat cos() throws GpuException
	{
		Thrust.cos(this); return this;
	}
	
	/**
	 * sin(a * x + b)
	 */
	public FloatMat sin(float a, float b) throws GpuException
	{
		Thrust.sin(this, a, b); return this;
	}
	public FloatMat sin() throws GpuException
	{
		Thrust.sin(this); return this;
	}
	
	/**
	 * sqrt(a * x + b)
	 */
	public FloatMat sqrt(float a, float b) throws GpuException
	{
		Thrust.sqrt(this, a, b); return this;
	}
	public FloatMat sqrt() throws GpuException
	{
		Thrust.sqrt(this); return this;
	}
	
	/**
	 * (a * x + b) ^p
	 */
	public FloatMat pow(float p, float a, float b) throws GpuException
	{
		Thrust.pow(this, p, a, b); return this;
	}
	public FloatMat pow(float p) throws GpuException
	{
		Thrust.pow(this, p); return this;
	}
	
	/**
	 * (a * x + b)
	 */
	public FloatMat linear(float a, float b) throws GpuException
	{
		Thrust.linear(this, a, b); return this;
	}
	
	public float max() throws GpuException
	{
		return Thrust.max(this);
	}

	public float min() throws GpuException
	{
		return Thrust.min(this);
	}

	public float sum() throws GpuException
	{
		return Thrust.sum(this);
	}

	public float product() throws GpuException
	{
		return Thrust.product(this);
	}
	
	public FloatMat sort() throws GpuException
	{
		Thrust.sort(this);	return this;
	}
	
	public FloatMat fill(float val) throws GpuException
	{
		Thrust.fill(this, val);	return this;
	}
	
	public FloatMat copyFrom(FloatMat other) throws GpuException
	{
		Thrust.copy(other, this);	return this;
	}
}
