package test.demo;

import gpu.FloatMat;

import java.io.*;
import utils.*;

public class SerializationDemo
{
	static class Dummy implements Serializable
	{
		private static final long serialVersionUID = 101L;
		private transient FloatMat mat;
		
		// make a 3-row mat 
		public Dummy(float[] arr)
		{
			this.mat = new FloatMat(arr, 3, arr.length / 3);
		}
		
		/**
		 * Must have EXACTLY the same signature for JVM to serialize
		 */
		private void writeObject(ObjectOutputStream out) throws IOException
		{
			out.defaultWriteObject();
			out.writeObject(mat.deflatten());
		}
		
		/**
		 * Must have EXACTLY the same signature for JVM to serialize
		 */
		private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException
		{
			in.defaultReadObject();
			this.mat = new FloatMat((float[][]) in.readObject());
		}
		
		public String toString()
		{
			return "Dummy " + this.mat.toString();
		}
	}
	
	static String fileName = "serialize_demo.dat";
	
	public static void main(String[] args) throws IOException
	{
		Dummy dum = new Dummy(new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
		FileUtil.dump(dum, fileName);
		PP.p("original:\n", dum);
		PP.p("Dump to file", fileName);
		PP.p("Load from file");
		dum = FileUtil.load(fileName);
		PP.p("loaded back:\n", dum);
		
		FileUtil.delete(fileName);
		PP.p("Deleted", fileName);
	}

}
