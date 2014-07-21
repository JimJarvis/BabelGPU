package test.deep;

import java.util.ArrayList;

import org.junit.*;

import utils.*;
import deep.*;
import deep.units.*;

public class LearningTest
{
	static InletUnit inlet;
	static LearningPlan plan;
	static int batch = 7;
	static int totalSamples = 24;
	static int totalEpochs = 5;
	static float[] losses = new float[] {2, 5, 1, 4, 3}; // specify loss manually
	
	@BeforeClass
	public static void setUp()
	{ 
		DeepTestKit.systemInit();
		
		plan = new LearningPlan("plan", "", 10, 1, totalSamples, totalEpochs);
		
		inlet = new InletUnit("inlet", 3, batch, true)
		{
			@Override
			protected int nextBatch_()
			{
				return Math.min(batch, plan.remainTrainSize());
			}

			@Override
			public void nextGold() { }

			@Override
			public void prepareNextEpoch() { }
		};
	}
	
	public static class MyTUnit extends TerminalUnit
	{
		public MyTUnit(String name, InletUnit inlet)
		{
			super(name, inlet);
		}

		public MyTUnit(String name, InletUnit inlet, boolean hasBias)
		{
			super(name, inlet, hasBias);
		}
		
		// DEBUGGING
		public ArrayList<Float> lrs = new ArrayList<>(); // lr each batch, every epoch
		public void clear() { lrs.clear(); } 

		@Override
		protected float forward_terminal(boolean doesCalcLoss)
		{
			PP.p("Epoch #", plan.curEpoch, "lr =", plan.lr);
			return losses[plan.curEpoch];
		}

		@Override
		public void backward() { }
	};
	
	@Test
//	@Ignore
	public void lrTest()
	{
		DeepNet net = 
				DeepFactory.debugLinearLayers(inlet, 
						new int[] {2, 1, 2}, 
						MyTUnit.class,
						Initializer.fillIniter(.1f));
		plan.setLrScheme(LrScheme.constantDecayScheme());
		net.run(plan);
		PP.p(plan.record);
		
		PP.pSectionLine();
		net.reset();
		plan.setLrScheme(new LrScheme()
		{
			@Override
			public float updateEpoch_(LearningPlan plan)
			{
				return -.5f * plan.lr;
			}
			@Override
			public float updateBatch_(LearningPlan plan)
			{
				return defaultLr();
			}
		});
		net.run(plan);
		PP.p(plan.record);
	}
}