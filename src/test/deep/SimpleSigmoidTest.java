package test.deep;

import deep.*;
import deep.units.*;

public class SimpleSigmoidTest
{

	public static void main(String[] args)
	{
		
		InletUnit inlet = new InletUnit("Dummy Inlet", 4, 1)
		{
			boolean hasNext = true;
			@Override
			public void nextGold()
			{
			}
			
			@Override
			public void nextBatch()
			{
				
				hasNext = false;
			}
			
			@Override
			public boolean hasNext()
			{
				return hasNext;
			}
		};
		DeepNet sigmoidNet = DeepFactory.simpleSigmoidNet(inlet, new int[] {5, 3});
		
	}

}
