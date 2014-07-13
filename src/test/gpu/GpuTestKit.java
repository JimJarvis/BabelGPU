package test.gpu;

import gpu.*;
import utils.*;

public class GpuTestKit
{

	/**
	 * Check the gold standard generated from Matlab
	 * Assume the goldFile has extension ".txt" and reside in bin/matlab_test
	 * @param testName
	 * @param tol tolerance of error
	 */
	public static void checkGold(FloatMat gpu, String goldFile, String testName, float tol)
	{
		CsvReader csv = new CsvReader("matlab_test/" + goldFile + ".txt");
		float[][] Gold = csv.readFloatMat();
		float[][] Host = gpu.deflatten();
		
		float diff = CpuUtil.matAvgDiff(Gold, Host);
		PP.setPrecision(3);
		PP.setScientific(true);
		
		PP.p("["+testName+"]", diff < tol ? "PASS:" : "FAIL:", diff);
	}
}
