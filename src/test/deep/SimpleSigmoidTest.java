package test.deep;

import java.util.HashMap;

import utils.*;
import gpu.*;
import deep.*;
import deep.units.*;

public class SimpleSigmoidTest
{
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
		
		int dim = dummyInput.length;
		int batchSize = dummyInput[0].length;
		
		InletUnit inlet = new InletUnit("Dummy Inlet", dim, batchSize)
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
		
		// totalTrainSize = colDim(input)
		LearningPlan plan = new LearningPlan(2, 1.5f, 0, batchSize);
		/**
		 * Simple forward sigmoid NN
		 */
		DeepNet sigmoidNet = DeepFactory.simpleSigmoidNet(inlet, new int[] {5, 20, 3, dim});
//		sigmoidNet.runDebug(plan);
//		sigmoidNet.gradCheck(plan);
		
		PP.pSectionLine("\n", 3);
		DeepNet linearLayers = DeepFactory.debugLinearLayers(inlet, new int[] {3, 5, 2, 1, 4, dim});
//		linearLayers.runDebug(plan);
//		linearLayers.gradCheck(plan);
		
		DeepNet sigmoidLayers = DeepFactory.debugSigmoidLayers(inlet, 1);
//		sigmoidLayers.runDebug(plan);
		sigmoidLayers.gradCheck(plan);
	}

}
