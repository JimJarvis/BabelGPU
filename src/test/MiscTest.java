package test;

import java.util.ArrayList;
import java.util.Random;

import utils.CpuUtil;
import utils.PP;

public class MiscTest
{
	public static void main(String[] args)
	{
		int r = 6;
		int c = 4;
		float[] randProjMatrix = new float[r * c];
		
		for(int i = 0; i < 6; i++)
		{
			for(int j = 0; j < c - 1; j++)
			{
				randProjMatrix[j*r + i] = (float)  6;//(rand.nextGaussian() * wScalingFactor);
			}
			randProjMatrix[(c - 1)*r + i] = (float) 3; // (rand.nextDouble() * 2.0 * Math.PI);
		}
			for(int j = 0; j < c - 1; j++)
			{
				randProjMatrix[j*r + r-1] = 0;
			}
			randProjMatrix[(c - 1)*r + r-1] = 1;
		
			PP.setSep("\n");
			PP.pMat(CpuUtil.deflatten(randProjMatrix, r, true));
	}

}
