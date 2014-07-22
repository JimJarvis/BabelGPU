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
	static float[] losses = new float[] {2, 5, 1, 4, 3}; // specify loss manually
	static int totalEpochs = losses.length;
	static DeepNet net;

	final static float INF = Float.POSITIVE_INFINITY;
	
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
		
		net = 
				DeepFactory.debugLinearLayers(inlet, 
						new int[] {2, 1, 2}, 
						MyTUnit.class,
						Initializer.fillIniter(.1f));
		plan.setLrScheme(LrScheme.constantDecayScheme());
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
	
	// After every test
	@After
	public void resetNet()
	{
		net.reset();
		losses = new float[] {2, 5, 1, 4, 3};
		plan.totalEpochs = totalEpochs = losses.length;
	}
	
	@Test
//	@Ignore
	public void constantDecayTest()
	{
		PP.pTitledSectionLine("Constant Decay");
		plan.setLrScheme(LrScheme.constantDecayScheme());
		net.run(plan);
		PP.p(plan.record);
	}
	
	@Test
	public void epochDummyUpdateTest()
	{
		PP.pTitledSectionLine("Epoch Dummy Update");
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
	
	@Test
	public void epochDecayTest()
	{
		PP.pTitledSectionLine("Epoch Decay");
		float decayRate = .5f;
		losses = new float[] 
				{5, 3, 2.9f, 2.7f, 2f, INF};  plan.totalEpochs = totalEpochs = losses.length;
		plan.setLrScheme(LrScheme.epochDecayScheme(0.2f, decayRate));
		net.run(plan);
		PP.p(plan.record);
		PP.p("\nCase 2\n");
		net.reset();
		losses = new float[] 
				{INF, INF, 5, 3, 2, 6, 5}; plan.totalEpochs = totalEpochs = losses.length;
		plan.setLrScheme(LrScheme.epochDecayScheme(0.2f, decayRate));
		net.run(plan);
		PP.p(plan.record);
		PP.p("\nCase 3\n");
		net.reset();
		losses = new float[] 
				{1, 2, INF, INF, 3, 2.9f, INF, 5, 4.9f}; plan.totalEpochs = totalEpochs = losses.length;
		plan.setLrScheme(LrScheme.epochDecayScheme(0.2f, decayRate));
		net.run(plan);
		PP.p(plan.record);
	}
}