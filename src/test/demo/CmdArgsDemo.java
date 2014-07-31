package test.demo;

import utils.MiscUtil;
import utils.PP;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

/**
 * Demo JCommander
 */
public class CmdArgsDemo
{
	public static class CmdArgs
	{
		@Parameter(names = {"-int", "-i"}, description = "Enter an int", required = true)
		public int i = 66;

		@Parameter(names = {"-bool", "-b"}, description = "Enter a boolean", required = true)
		public boolean b = false;
		
		@Parameter(names = "-str", description = "dudulu")
		public String s = "default";
		
		@Parameter(names = "-intarr", description = "int array", converter = MiscUtil.IntArrayConverter.class)
		public int[] iarr;

		@Parameter(names = "-floatarr", description = "float array", converter = MiscUtil.FloatArrayConverter.class)
		public float[] farr;
		
		@Parameter(names = "-doublemat", description = "double matrix", converter = MiscUtil.DoubleMatConverter.class)
		public double[][] dmat;
		
		@Parameter(names = "-strmat", description = "string matrix", converter = MiscUtil.StringMatConverter.class)
		public String[][] smat;
		
		@Parameter(names = "--help", help = true)
		public boolean help = false;
		
		public CmdArgs() { }
		public CmdArgs(int i, boolean b)
		{
			this.i = i;
			this.b = b;
		}
	}

	public static void main(String[] args)
	{
		args = new String[] {
				"-int", "3", 
				"-str", "dudulu", 
				"-bool", 
				"-intarr", "3,9,20,-5", 
				"-floatarr", "-0.3,1.21,4.9e-3,-.6e5",
				"-doublemat", "1.2,-3.4,,5.6,7.8,,-9.0,2.3",
				"-strmat", "lap,gauss,cauchy,,mid,,a,b,cc,ddd",
				"--help"};
		CmdArgs cmd = new CmdArgs();
		JCommander jcmder = new JCommander(cmd, args);
		
		if (cmd.help)
			jcmder.usage();
		PP.setSep("\n");
		PP.p(cmd.i, cmd.b, cmd.s, cmd.iarr, cmd.farr, cmd.dmat, cmd.smat);
	}

}
