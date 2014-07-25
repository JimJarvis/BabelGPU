package test;

import java.lang.reflect.Array;
import java.util.*;

import deep.DeepNet;
import utils.FileUtil.Writer;
import utils.MiscUtil.*;
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
		
//		for (String line : FileUtil.iterable("../BabelGPU", "test.sh")) PP.p(line);
		
		PP.p(MiscUtil.splitStrNum("Z_a-3.4"));
		PP.p(MiscUtil.splitStrNum("Z_a"));
		
		Integer[] a = new Integer[] {3, null, 5, 6};
		a = new Integer[] {8, null, null, null, null};
		String[] s = new String[] {"aa", "bbb", "c", "dddd"};
        s = new String[] {null, null, null, null, "dud"};
		Pair<Integer, String>[] p = Pair.unzip(new Pair<>(a, s));
		PP.po(p);
		Pair<Integer[], String[]> zipped = Pair.zip(p);
		PP.po(zipped.o1[0]);
		
		String A[] = new String[] {"lap2.3", "j_a-4", "_AZ63.2", "dud-2.4e3"};
		Pair<String, Double> psd[] = MiscUtil.map(A, new DualFunc<String, Pair<String, Double>>()
				{
					@Override
					public Pair<String, Double> apply(String in)
					{
						return MiscUtil.splitStrNum(in);
					}
				}, Pair.class);
		PP.po(psd);
		
		PP.p(CpuUtil.cumultProduct(new int[] {3, 5, 10, 7}));
		
		PP.p("DONE");
	}
}
