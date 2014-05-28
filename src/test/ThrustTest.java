package test;

import utils.PP;
import gpu.*;

public class ThrustTest
{

	public static void main(String[] args) throws GpuException
	{
		float[] f = new float[] {25, 100, 16, 1024};
		FloatMat O = new FloatMat(f);
		
		FloatMat a = new FloatMat(O.numRows, O.numCols, true /*memsetToZero*/);
		a.fill(66);
		PP.p(a);
		
		a.copyFrom(O); a.pow(0.5f); PP.p(a);
		a.copyFrom(O); a.pow(0.5f, 0.7f, 0); PP.p(a);
		a.copyFrom(O); a.pow(0.5f, 1, 4); PP.p(a);
		a.copyFrom(O); a.pow(0.5f, .7f, 4); PP.p(a);

		a.copyFrom(O); a.sqrt(); PP.p(a);
		a.copyFrom(O); a.sqrt( 0.7f, 0); PP.p(a);
		a.copyFrom(O); a.sqrt( 1, 4); PP.p(a);
		a.copyFrom(O); a.sqrt( .7f, 4); PP.p(a);
		
		O.destroy(); a.destroy();

		O = new FloatMat(new float[] {4.2f, 5.9f, -2.1f, -3.7f, 3.3f, 1.9f, -0.6f});
		a = new FloatMat(O.numRows, O.numCols, true /*memsetToZero*/);
		a.copyFrom(O); Thrust.babel_id_minus_softmax(a, 3); PP.p(a);
	}

}
