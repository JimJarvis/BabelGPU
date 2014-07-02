package test.demo;

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
		@Parameter(names = "-int", description = "Enter an int")
		public int i = 66;
		
		@Parameter(names = "-bool", description = "Enter a boolean")
		public boolean b = false;
		
		@Parameter(names = "-str", description = "dudulu")
		public String s = "default";
		
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
		args = new String[] {"-int", "3", "-str", "dudulu", "-bool", "--help"};
		CmdArgs cmd = new CmdArgs();
		JCommander jcmder = new JCommander(cmd, args);
		
		if (cmd.help)
			jcmder.usage();
		PP.p(cmd.i, cmd.b, cmd.s, cmd.help);
	}

}
