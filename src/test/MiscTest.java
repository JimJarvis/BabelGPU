package test;

import java.util.ArrayList;

import utils.CpuUtil;
import utils.PP;

public class MiscTest
{
	public static void main(String[] args)
	{
		Integer a = 6;
		Integer b[] = CpuUtil.repeatedArray(a, 8);
		PP.po(b);
		
		ArrayList<Integer> A = new ArrayList<>();
		A.add(6);
		A.add(66);
		ArrayList<Integer> B = (ArrayList<Integer>) A.clone();
		B.add(-100);
		
		PP.p(A, B);
	}

}
