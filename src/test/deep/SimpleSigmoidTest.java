package test.deep;

import java.util.HashMap;

import utils.*;
import gpu.*;
import deep.*;
import deep.units.*;

public class SimpleSigmoidTest
{
	private static DeepNet sigmoidNet;
    private static HashMap<String, ComputeUnit> unitMap;
	
	public static void main(String[] args)
	{
		GpuBlas.init();
		
		final float[][] dummyInput = new float[][] {
				{1.5f, 3},
				{-0.5f, -1},
				{-2, -4},
				{3, 6}
		};

		final float[][] dummyGold = new float[][] {
				{0, 0},
				{1, 0},
				{0, 1},
				{0, 0}
		};
		
		InletUnit inlet = new InletUnit("Dummy Inlet", 4, 2)
		{
			boolean hasNext = true;
			{
				this.goldMat = new FloatMat(this.data);
			}
			
			@Override
			public void nextGold()
			{
				this.goldMat.setHostArray(dummyGold);
				this.goldMat.toDevice(true);
			}
			
			@Override
			public void nextBatch()
			{
				this.data.setHostArray(dummyInput);
				this.data.toDevice(true);
				hasNext = false;
			}
			
			@Override
			public boolean hasNext()
			{
				return hasNext;
			}
		};
		sigmoidNet = DeepFactory.simpleSigmoidNet(inlet, new int[] {5, 3});
		unitMap = sigmoidNet.getUnitMap();
		LearningPlan plan = new LearningPlan(1, 1, 0, dummyInput.length);
		sigmoidNet.runDebug(plan);
		
	}

}
