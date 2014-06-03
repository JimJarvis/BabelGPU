package test.demo;

import utils.PP;
import edu.stanford.nlp.optimization.*;

public class LbfgsDemo
{

	public static void main(String[] args)
	{
		QNMinimizer qm = new QNMinimizer();
		qm.shutUp();
		DiffFunction dffFloat = new DiffFunction()
		{
			@Override
			public double valueAt(double[] x)
			{
				return x[0] * x[0] - 4*x[0] + 4 + x[1] * x[1];
			}
			
			@Override
			public int domainDimension()
			{
				return 2;
			}
			
			@Override
			public double[] derivativeAt(double[] x)
			{
				return new double[] {x[0]*2 - 4, x[1]*2};
			}
		};
		
		double[] initguess = new double[] {-1.3f, 2.5f};
		qm.minimize(dffFloat, 1e-5, initguess);
		
		PP.p(initguess);
	}

}
