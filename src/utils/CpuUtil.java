package utils;

public class CpuUtil
{
	// ******************** COMMON ********************/
	
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
	
	//**************************************************/
	//******************* FLOAT *******************/
	//**************************************************/
	/**
	 * Utility: deflatten a 1D float to 2D matrix
	 * @param row row dimension
	 * @param columnMajor true for column major, false for row major
	 */
	public static float[][] deflatten(float[] A, int row, boolean columnMajor)
	{
		float[][] out = new float[row][A.length / row];
		return deflatten(A, out, columnMajor);
	}
	
	/**
	 * Utility: deflatten a 1D float to 2D matrix
	 * @param out output parameter
	 * @param columnMajor true for column major, false for row major
	 */
	public static float[][] deflatten(float[] A, float[][] out, boolean columnMajor)
	{
		int row = out.length;
		int col = out[0].length;
		int pt = 0;
		
		if (columnMajor)
    		for (int j = 0; j < col; j ++)
    			for (int i = 0; i < row; i ++)
    				out[i][j] = A[pt ++];
		else // row major
			for (int i = 0; i < row; i ++)
				for (int j = 0; j < col; j ++)
					out[i][j] = A[pt ++];
		return out;
	}
	
	/**
	 * Utility: flatten a 2D float array to 1D
	 * @param out output parameter
	 * @param columnMajor true for column major, false for row major
	 */
	public static float[] flatten(float[][] A, float[] out, boolean columnMajor)
	{
		int row = A.length;
		int col = A[0].length;
		int pt = 0;

		if (columnMajor)
    		for (int j = 0; j < col; j ++)
    			for (int i = 0; i < row; i ++)
    				out[pt ++] = A[i][j];
		else
			for (int i = 0; i < row; i ++)
				for (int j = 0; j < col; j ++)
					out[pt ++] = A[i][j];

		return out;
	}
	
	/**
	 * Utility: flatten a 2D float array to 1D
	 * @param columnMajor true for column major, false for row major
	 */
	public static float[] flatten(float[][] A, boolean columnMajor)
	{
		float[] out = new float[A.length * A[0].length];
		return flatten(A, out, columnMajor);
	}
	
	/**
	 * Returns the average of absolute difference from 2 matrices
	 */
	public static float matAvgDiff(float[][] A, float[][] B)
	{
		float diff = 0;
		int r = A.length;
		int c = A[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j ++)
				diff += Math.abs(A[i][j] - B[i][j]);
		
		return diff / (r * c);
	}
	
	public static float[][] mult(float[][] a, float[][] b)
	{
		float[][] c = new float[a.length][b[0].length];
		for(int i=0;i<a.length;i++){
	        for(int j=0;j<b[0].length;j++){
	            for(int k=0;k<b.length;k++){
	            c[i][j]+= a[i][k] * b[k][j];
	            }
	        }
	    }
		return c;
	}
	
	public static float[][] addCol1(float[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		float[][] b = new float[r][c+1];
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				b[i][j] = a[i][j];

		for (int i = 0; i < r; i++)
			b[i][c] = 1;
		
		return b;
	}
	
	public static float[][] transpose(float[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		float[][] t = new float[c][r];
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				t[j][i] = a[i][j];
		return t;
	}
	
	public static float[][] cos(float[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				a[i][j] = (float) Math.cos(a[i][j]);
		return a;
	}
	
	public static float[][] getCol(float[][] a, int c)
	{
		int r = a.length;
		float[][] col = new float[r][1];
		for (int i = 0; i < r; i++)
			col[i][0] = a[i][c];
		return col;
	}
	
	public static float sum(float[][] a)
	{
		float s = 0;
		int r = a.length;
		int c = a[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				s += a[i][j];
		return s;
	}
	
	public static void float2double(float[] in, double[] out)
	{
		for (int i = 0 ; i < in.length ; i++)
			out[i] = (double) in[i];
	}
	
	//**************************************************/
	//******************* DOUBLE *******************/
	//**************************************************/
	/**
	 * Utility: deflatten a 1D double to 2D matrix
	 * @param row row dimension
	 * @param columnMajor true for column major, false for row major
	 */
	public static double[][] deflatten(double[] A, int row, boolean columnMajor)
	{
		double[][] out = new double[row][A.length / row];
		return deflatten(A, out, columnMajor);
	}
	
	/**
	 * Utility: deflatten a 1D double to 2D matrix
	 * @param out output parameter
	 * @param columnMajor true for column major, false for row major
	 */
	public static double[][] deflatten(double[] A, double[][] out, boolean columnMajor)
	{
		int row = out.length;
		int col = out[0].length;
		int pt = 0;
		
		if (columnMajor)
    		for (int j = 0; j < col; j ++)
    			for (int i = 0; i < row; i ++)
    				out[i][j] = A[pt ++];
		else // row major
			for (int i = 0; i < row; i ++)
				for (int j = 0; j < col; j ++)
					out[i][j] = A[pt ++];
		return out;
	}
	
	/**
	 * Utility: flatten a 2D double array to 1D
	 * @param out output parameter
	 * @param columnMajor true for column major, false for row major
	 */
	public static double[] flatten(double[][] A, double[] out, boolean columnMajor)
	{
		int row = A.length;
		int col = A[0].length;
		int pt = 0;

		if (columnMajor)
    		for (int j = 0; j < col; j ++)
    			for (int i = 0; i < row; i ++)
    				out[pt ++] = A[i][j];
		else
			for (int i = 0; i < row; i ++)
				for (int j = 0; j < col; j ++)
					out[pt ++] = A[i][j];

		return out;
	}
	
	/**
	 * Utility: flatten a 2D double array to 1D
	 * @param columnMajor true for column major, false for row major
	 */
	public static double[] flatten(double[][] A, boolean columnMajor)
	{
		double[] out = new double[A.length * A[0].length];
		return flatten(A, out, columnMajor);
	}
	
	/**
	 * Returns the average of absolute difference from 2 matrices
	 */
	public static double matAvgDiff(double[][] A, double[][] B)
	{
		double diff = 0;
		int r = A.length;
		int c = A[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j ++)
				diff += Math.abs(A[i][j] - B[i][j]);
		
		return diff / (r * c);
	}
	
	public static double[][] mult(double[][] a, double[][] b)
	{
		double[][] c = new double[a.length][b[0].length];
		for(int i=0;i<a.length;i++){
	        for(int j=0;j<b[0].length;j++){
	            for(int k=0;k<b.length;k++){
	            c[i][j]+= a[i][k] * b[k][j];
	            }
	        }
	    }
		return c;
	}
	
	public static double[][] addCol1(double[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		double[][] b = new double[r][c+1];
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				b[i][j] = a[i][j];

		for (int i = 0; i < r; i++)
			b[i][c] = 1;
		
		return b;
	}
	
	public static double[][] transpose(double[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		double[][] t = new double[c][r];
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				t[j][i] = a[i][j];
		return t;
	}
	
	public static double[][] cos(double[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				a[i][j] = (double) Math.cos(a[i][j]);
		return a;
	}
	
	public static double[][] getCol(double[][] a, int c)
	{
		int r = a.length;
		double[][] col = new double[r][1];
		for (int i = 0; i < r; i++)
			col[i][0] = a[i][c];
		return col;
	}
	
	public static double sum(double[][] a)
	{
		double s = 0;
		int r = a.length;
		int c = a[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				s += a[i][j];
		return s;
	}
	
	public static void double2float(double[] in, float[] out)
	{
		for (int i = 0 ; i < in.length ; i++)
			out[i] = (float) in[i];
	}
}
