package test.gpu;

import static org.junit.Assert.*;
import com.googlecode.javacpp.IntPointer;
import gpu.*;
import utils.*;

public class GpuTestKit
{
	private String sessionName;
	public static float TOL = 1e-9f;
	private CsvReader csv;
	
	public GpuTestKit(String sessionName)
	{
		this.sessionName = sessionName;
	}

	public static void systemInit()
	{
		GpuBlas.init();
		GpuUtil.enableExceptions();
		PP.setPrecision(3);
		PP.setScientific(true);
	}
	
	/**
	 * Always in the folder 'matlab_test/' which locates in 'bin/'
	 */
	private void newReader(String file)
	{
		csv = new CsvReader("matlab_test/" + sessionName + "_" + file + ".txt");
	}

	public int[] loadInts(String file)
	{
		newReader(file);
		return csv.readIntVec(true);
	}
	
	/**
	 * Load int array, say 'labels', directly to GPU
	 */
	public IntPointer loadIntGpu(String file)
	{
		return Thrust.copy_host_to_device(loadInts(file));
	}
	
	public FloatMat loadFloatMat(String file, int row, int col)
	{
		newReader(file);
		FloatMat x = new FloatMat(csv.readFloatVec(true), row, col);
		x.toDevice(true);
		return x;
	}
	
	public float[] loadFloats(String file)
	{
		newReader(file);
		return csv.readFloatVec(true);
	}
	
	/**
	 * Malloc an int array on GPU
	 */
	public static IntPointer createIntGpu(int size)
	{
		return Thrust.malloc_device_int(size);
	}
	
	private static float matDiff(FloatMat x, FloatMat gold)
	{
		FloatMat diffMat = new FloatMat(x);
		GpuBlas.add(x, gold, diffMat , 1, -1);
		float diff = diffMat .abs().sum() / diffMat .size();
		diffMat.destroy();
		return diff;
	}

	/**
	 * Check the gold standard generated from Matlab
	 * Assume the goldFile has extension ".txt" and reside in bin/matlab_test
	 * @param goldFile must be prefixed with 'gold_'
	 * @param tol tolerance of error
	 */
	public void checkGold(FloatMat gpu, String goldFile, String description)
	{
		newReader(goldFile);
		float diff = matDiff(gpu, new FloatMat(csv.readFloatMat()));
		PP.p("["+description+"]", diff < TOL ? "PASS:" : "FAIL:", diff);
		assertTrue("Doesn't agree with Gold", diff < TOL);
	}
	public void checkGold(FloatMat gpu, String goldFile)
	{
		checkGold(gpu, goldFile, goldFile.substring(5));
	}
	
	/**
	 * Simply compare two numbers. The files contains one number
	 */
	public void checkGold(float res, String goldFile, float tol, String description)
	{
		if (tol < 0) tol = TOL;
		newReader(goldFile);
		float gold = csv.readFloatVec(1)[0];
		float diff = Math.abs(gold - res);
		PP.p("["+description+"]", diff < tol ? "PASS" : "FAIL");
		if (diff > tol)
    	{
			PP.p("yours=", res, "  but gold=", gold);
			fail("Doesn't agree with Gold");
    	}
	}
	
	/**
	 * Compare two FloatMats
	 */
	public void checkGold(FloatMat x, FloatMat gold, String description)
	{
		float diff = matDiff(x, gold);
		PP.p("["+description+"]", diff < TOL ? "PASS:" : "FAIL:", diff);
		assertTrue("Doesn't agree with Gold", diff < TOL);
	}
	
	public void printSessionTitle()
	{
		PP.pTitledSectionLine(sessionName + " Test", "%", 20);
	}
}
