package test.deep;

import gpu.FloatMat;
import gpu.FloatMat.Saveable;
import gpu.GpuRand;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import org.junit.*;

import utils.*;
import deep.*;
import deep.units.*;

/**
 * Test serialization: save net to file
 */
public class SaveTest
{
	static InletUnit inlet;
	static LearningPlan plan;
	static int batch = 7;
	static int totalSamples = 24;
	static float[] losses = new float[] {2, 5, 1, 4, 3}; // specify loss manually
	static int totalEpochs = losses.length;
	static DeepNet net;
	static final int SAVE_BOTH = DataUnit.SAVE_DATA | DataUnit.SAVE_GRADIENT;

	final static float INF = Float.POSITIVE_INFINITY;
	
	@BeforeClass
	public static void setUp()
	{ 
		DeepTestKit.systemInit();
		
		final int dim = 3;
		plan = new LearningPlan("plan", "", 10, 1, totalSamples, totalEpochs);
		
		inlet = new InletUnit("inlet", dim, batch, true)
		{
			private static final long serialVersionUID = 1L;
			transient GpuRand grand;
			transient FloatMat data_;
			final String inletPath = "inlet_data.float";
			@Override
			protected int nextBatch_()
			{
				if (grand == null)
					 grand = new GpuRand();
				int b = Math.min(batch, getPlan().remainTrainSize());
				if (data_ == null)
					data_ = grand.genUniformFloat(dim, b);
				this.data = data_;
				return b;
			}

			@Override
			public void nextGold() { }

			@Override
			public void prepareNextEpoch() { }
			
			private void writeObject(ObjectOutputStream out) throws IOException
			{
				out.defaultWriteObject();
				out.writeObject(FloatMat.saveable(data_, inletPath));
			}
			
			private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException
			{
				in.defaultReadObject();
				this.data_ = FloatMat.desaveable( (Saveable) in.readObject());
			}
		};
		
		net = 
				DeepFactory.debugLinearLayers(inlet, 
						new int[] {2, 3, 2}, 
						MyTUnit.class,
						Initializer.uniformRandIniter(1f));
		plan.setLrScheme(LrScheme.constantDecayScheme());
	}
	
	public static class MyTUnit extends TerminalUnit
	{
		private static final long serialVersionUID = 1L;
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
			PP.p("Epoch #", getPlan().curEpoch, "lr =", getPlan().lr);
			return losses[getPlan().curEpoch];
		}

		@Override
		public void backward() { }
	};
	
	@Test
//	@Ignore
	public void saveTest()
	{
//		net.enableDebug(true);
		net.setUnitOutputSaveMode(DataUnit.SAVE_DATA);
		net.setParamSaveMode(SAVE_BOTH);
		net.run(plan);
		PP.p("OUTPUT units");
		for (ComputeUnit cu : net)
			PP.p(cu.output);
		PP.p("PARAM_LIST\n", net.paramList);
		PP.p("BEST_PARAM_LIST\n", net.bestParamList);
		PP.p("record\n", net.learningPlan.record);
		FileUtil.dump(net, "net.dat");
		PP.pTitledSectionLine("Save done");

		PP.pSectionLine("=", 70);
		DeepNet forked = FileUtil.<DeepNet>load("net.dat");
		PP.pTitledSectionLine("Load done");
//		forked.learningPlan.reset();
//		forked.reset();
//		forked.run(forked.learningPlan);
		PP.p("OUTPUT units");
		for (ComputeUnit cu : forked)
			PP.p(cu.output);
		PP.p("PARAM_LIST\n", forked.paramList);
		PP.p("BEST_PARAM_LIST\n", forked.bestParamList);
		PP.p("record\n", forked.learningPlan.record);
		forked.run(forked.learningPlan);
	}
}