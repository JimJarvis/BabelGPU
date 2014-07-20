package utils;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Miscellaneous util methods. 
 */
public class MiscUtil
{
	/**
	 * Make a repeated array from a single object
	 */
	public static <T> T[] repeatedArray(T obj, int len)
	{
		T[] objs = (T[]) Array.newInstance(obj.getClass(), len);
		for (int i = 0 ; i < len ; i ++)
			objs[i] = obj;
		return objs;
	}
	
	/**
	 * Make a repeated array of floats
	 */
	public static float[] repeatedArray(float f, int len)
	{
		float[] ans = new float[len];
		for (int i = 0; i < len; ++ i)		ans[i] = f;
		return ans;
	}
	
	/**
	 * @return ArrayList wrapper around this array
	 */
	public static <T> ArrayList<T> toList(T ... arr) 
	{
		return new ArrayList<T>(Arrays.asList(arr));
	}
	
	/**
	 * @return primitive array
	 */
	public static <T> T[] toArray(ArrayList<T> list)
	{
		T[] arr = (T[]) new Object[list.size()];
		list.toArray(arr);
		return arr;
	}
	
	/**
	 * Inner class for 2D coordinate in the matrix
	 */
	public static class Coord
	{
		public int i; // row
		public int j; // col
		public Coord(int i, int j)
		{
			this.i = i; 
			this.j = j;
		}
		public String toString() { return String.format("<%d, %d>", i, j); }
	}
	
	/**
	 * Transform an index to a coordinate (column major)
	 */
	public static Coord toCoord(int row, int idx)
	{
		return new Coord(idx%row, idx/row);
	}
	
	/**
	 * Transform a 2D coordinate to index (column major)
	 */
	public static int toIndex(int row, int i, int j)
	{
		return j * row + i;
	}
	/**
	 * Transform a 2D coordinate to index (column major)
	 */
	public static int toIndex(int row, Coord c)
	{
		return c.j * row + c.i;
	}
}
