package test;

import utils.CpuUtil;
import utils.PP;

public class MiscTest
{
	public static void main(String[] args)
	{
		Integer a = 6;
		Integer b[] = CpuUtil.repeatedArray(a, 8);
		PP.po(b);
	}

}
