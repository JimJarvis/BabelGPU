package utils;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import com.beust.jcommander.IStringConverter;

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
	public static <T> T[] toArray(ArrayList<T> list, Class cl)
	{
		T[] arr = (T[]) Array.newInstance(cl, list.size());
		list.toArray(arr);
		return arr;
	}
	
	/**
	 * Get element. Python style indexing
	 */
	public static <T> T get(ArrayList<T> list, int i)
	{
		if (i < 0)	i += list.size();
		return list.get(i);
	}
	
	/**
	 * Get element. Python style indexing
	 */
	public static <T> T get(T[] arr, int i)
	{
		if (i < 0)	i += arr.length;
		return arr[i];
	}
	/**
	 * Get element. Python style indexing
	 */
	public static int get(int[] arr, int i) { if (i < 0)	i += arr.length; return arr[i]; }
	
	public static <T> boolean isEmpty(T[] arr) { return arr.length == 0; }
	public static boolean isEmpty(int[] arr) { return arr.length == 0; }
	
	/**
	 * Set element. Python style indexing
	 */
	public static <T> void set(ArrayList<T> list, int i, T ele)
	{
		if (i < 0)	i += list.size();
		list.set(i, ele);
	}
	
	/**
	 * Set element. Python style indexing
	 */
	public static <T> void set(T[] arr, int i, T ele)
	{
		if (i < 0)	i += arr.length;
		arr[i] = ele;
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
	
	/**
	 * while (iter.hasNext_) { 
	 * 		iter.next();
	 * 		// for-each loop logic
	 * 		iter.trailer() // called at the end of the loop
	 * }
	 */
	public static abstract class ManagedIterator<T> implements Iterator<T>
	{
		private boolean first = true;
		
		public final boolean hasNext()
		{
			if (!first)
				trailer();
			first = false;
			return hasNext_();
		}
		
		public abstract boolean hasNext_();
		
		/**
		 * For-each loop do at first of each loop
		 */
		public abstract T next();

		/**
		 * For-each loop do at last of each loop
		 */
		public abstract void trailer();

		@Override
		public final void remove()
		{
			throw new UnsupportedOperationException(
					"ManagedIterator doesn't support remove()");
		}
	}
	
	/**
	 * Append to tail
	 */
	public static <T> T[] append(T[] arr, T last)
	{
	    final int N = arr.length;
	    arr = java.util.Arrays.copyOf(arr, N+1);
	    arr[N] = last;
	    return arr;
	}
	public static int[] append(int[] arr, int last)
	{
	    final int N = arr.length;
	    arr = java.util.Arrays.copyOf(arr, N+1);
	    arr[N] = last;
	    return arr;
	}
	
	/**
	 * Prepend to head
	 */
	public static <T> T[] prepend(T first, T[] arr)
	{
	    final int N = arr.length;
	    arr = java.util.Arrays.copyOf(arr, N+1);
	    System.arraycopy(arr, 0, arr, 1, N);
	    arr[0] = first;
	    return arr;
	}
	public static int[] prepend(int first, int[] arr)
	{
	    final int N = arr.length;
	    arr = java.util.Arrays.copyOf(arr, N+1);
	    System.arraycopy(arr, 0, arr, 1, N);
	    arr[0] = first;
	    return arr;
	}
	
	//**************************************************/
	//*********** Functional programming interface *******/
	//**************************************************/
	/**
	 * Input and Output have different types
	 */
	public static interface DualFunc<In, Out>
	{
		public Out apply(In in);
	}
	/**
	 * Supports index enumeration
	 */
	public static interface IndexedDualFunc<In, Out>
	{
		public Out apply(In in, int idx);
	}
	
	/**
	 * Supports index enumeration
	 */
	public static interface MonoFunc<T>
	{
		public T apply(T in);
	}
	/**
	 * Input and Output have the same types
	 */
	public static interface IndexedMonoFunc<T>
	{
		public T apply(T in, int idx);
	}
	
	/**
	 * Functional: map operation
	 */
	public static <In, Out> Out[] map(In[] input, DualFunc<In, Out> f, Class outClass)
	{
		Out[] ans = (Out[]) Array.newInstance(outClass, input.length);
		int i = 0;
		for (In in : input)
			ans[i ++] = f.apply(in);
		return ans;
	}
	/**
	 * Functional: map operation with index
	 */
	public static <In, Out> Out[] map(In[] input, IndexedDualFunc<In, Out> f, Class outClass)
	{
		Out[] ans = (Out[]) Array.newInstance(outClass, input.length);
		int i = 0;
		for (In in : input)
		{
			ans[i] = f.apply(in, i);
			++ i;
		}
		return ans;
	}

	/**
	 * Functional: map operation
	 */
	public static <T> T[] map(T[] input, MonoFunc<T> f)
	{
		T[] ans = (T[]) Array.newInstance(
				input.getClass().getComponentType(), input.length);
		int i = 0;
		for (T in : input)
			ans[i ++] = f.apply(in);
		return ans;
	}
	/**
	 * Functional: map operation
	 */
	public static <T> T[] map(T[] input, IndexedMonoFunc<T> f)
	{
		T[] ans = (T[]) Array.newInstance(
				input.getClass().getComponentType(), input.length);
		int i = 0;
		for (T in : input)
		{
			ans[i] = f.apply(in, i);
			++ i;
		}
		return ans;
	}
	
	//********** JCommander converter classes *************/
	/**
	 * Comma separated
	 */
	public static class IntArrayConverter implements IStringConverter<int[]>
	{
		@Override
		public int[] convert(String arg0)
		{
			String[] strs = arg0.split("[,]");
			int[] ints = new int[strs.length];
			int i = 0;
			for (String str : strs)
				ints[i++] = Integer.parseInt(str);
			return ints;
		}
	}
	/**
	 * Comma separated
	 */
	public static class FloatArrayConverter implements IStringConverter<float[]>
	{
		@Override
		public float[] convert(String arg0)
		{
			String[] strs = arg0.split("[,]");
			float[] floats = new float[strs.length];
			int i = 0;
			for (String str : strs)
				floats[i++] = Float.parseFloat(str);
			return floats;
		}
	}
	/**
	 * Comma separated
	 */
	public static class DoubleArrayConverter implements IStringConverter<double[]>
	{
		@Override
		public double[] convert(String arg0)
		{
			String[] strs = arg0.split("[,]");
			double[] doubles = new double[strs.length];
			int i = 0;
			for (String str : strs)
				doubles[i++] = Double.parseDouble(str);
			return doubles;
		}
	}
	/**
	 * Comma separated
	 */
	public static class StringArrayConverter implements IStringConverter<String[]>
	{
		@Override
		public String[] convert(String arg0) { return arg0.split("[,]"); }
	}
	
	/**
	 * Rows comma separated, cols double comma separated
	 */
	public static class StringMatConverter implements IStringConverter<String[][]>
	{
		@Override
		public String[][] convert(String arg0)
		{
			String[] rowstrs = arg0.split(",,");
			String[][] smat = new String[rowstrs.length][];
			int r = 0;
			for (String row : rowstrs)
				smat[r++] = row.split("[,]");
			return smat;
		}
	}
	
	/**
	 * Rows comma separated, cols double comma separated
	 */
	public static class DoubleMatConverter implements IStringConverter<double[][]>
	{
		@Override
		public double[][] convert(String arg0)
		{
			String[] rowstrs = arg0.split(",,");
			double[][] dmat = new double[rowstrs.length][];
			int c, r = 0;
			for (String row : rowstrs)
			{
				String[] elestrs = row.split("[,]");
				c = 0;
				dmat[r] = new double[elestrs.length];
				for (String ele : elestrs)
					dmat[r][c ++] = Double.parseDouble(ele);
				++ r;
			}
			return dmat;
		}
	}
	
	/**
	 * Given a string concat by a letter and a double, split them up. 
	 * E.g.  "lap3.45" split into "lap" and 3.45
	 * If the number doesn't exist, return null for 'Double' field
	 */
	public static Pair<String, Double> splitStrNum(String arg)
	{
		int s = 0; // split point
		char c;
		do
			if (s == arg.length()) // no number 
				return new Pair<>(arg, null);
			else
                c = arg.charAt(s++);
		while ('a' <= c && c <= 'z'
				|| 'A' <= c && c <= 'Z'
				|| c == '_');
		return new Pair<>(
				arg.substring(0, s-1), 
                Double.parseDouble(arg.substring(s-1)));
	}
}
