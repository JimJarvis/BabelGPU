package test;

import java.util.*;
import deep.DeepNet;
import utils.*;

public class MiscTest
{
	public static void main(String[] args)
	{
		
//			PP.setSep("\n");
//			PP.pMat(CpuUtil.deflatten(randProjMatrix, r, true));
//		PP.p(CpuUtil.randInts(20, 3, 8));
//		for (Path p : FileUtil.listDir("src", true)) PP.p(p);
		
//		FileUtil.move("shit/mover", FileUtil.join("shit6", "sub7"));
//		for (String p : FileUtil.listDir("shit6", "*.txt", true)) PP.p(p);
//		FileUtil.makeDir("kirito");
//		FileUtil.makeTempDir("kirito", "");
//		FileUtil.makeTempFile("kirito", ".txt", "");
		float lastLoss = Float.POSITIVE_INFINITY;
		lastLoss = 400;
		float curLoss = 399.59999f;
		float improvement = (lastLoss - curLoss) / lastLoss;
		boolean decay = Float.isNaN(improvement) || improvement < 0.001f;
		PP.p(Float.POSITIVE_INFINITY > 2);
//		for (String line : FileUtil.iterable("../BabelGPU", "test.sh")) PP.p(line);
		
		PP.p("DONE");
	}

}
