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
				{1.5f, 3, -.5f},
				{-0.5f, -1, 2},
				{-2, -4, 2.5f},
				{3, 6, 0}
		};

		final float[][] dummyGold = new float[][] {
				{0, 0, 1},
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 0}
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

			@Override
			public void reset()
			{
				hasNext = true;
			}
		};
		sigmoidNet = DeepFactory.debugSimpleSigmoidNet(inlet, new int[] {5, 3});
		unitMap = sigmoidNet.getUnitMap();
		// totalTrainSize = colDim(input)
		LearningPlan plan = new LearningPlan(1, 0, 0, dummyInput[0].length);
		
//		sigmoidNet.runDebug(plan);
//		PP.pSectionLine("\n", 3);
//		sigmoidNet.gradCheck(plan);
		
		DeepNet linearLayer = DeepFactory.debugLinearLayer(inlet, new int[] {3});
		linearLayer.runDebug(plan);
		linearLayer.reset();
		linearLayer.gradCheck(plan);
	}

}
