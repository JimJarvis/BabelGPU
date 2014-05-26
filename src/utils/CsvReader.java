package utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

import utils.Pickle;

/**
 * Read comma separated files into matrices/vectors. <br>
 * Each column is comma-separated. Each row occupies a new line. <br>
 * Supports three data types: int, float and String <br>
 * Supports read resuming
 * @author Jim Fan  (c) 2014
 */
public class CsvReader
{
	private BufferedReader reader; // can resume reading rows
	private Scanner colScanner; // can resume reading 1D array

	private String fileName;
	private String delimiter = ",";

	/**
	 * Construct a CsvReader from a file
	 * @param fileName
	 */
	public CsvReader(String fileName)
	{
		this.fileName = fileName;
		restart();
	}

	/**
	 * Change delimiter, default comma
	 */
	public void useDelimiter(String delimiter) {	this.delimiter = delimiter;	}

	/**
	 * Refresh the file reader and restart from the beginning
	 */
	public void restart()
	{
		try {
			reader = new BufferedReader(new FileReader(new File(fileName)));
			colScanner = null;
		}
		catch (FileNotFoundException e)
		{ System.err.println("File not found :" + fileName); }
	}

	/**
	 * Capable of resuming read
	 * @param row read a specified number of rows
	 * @param col read a specified number of columns
	 * @return an int matrix from CSV file
	 */
	public int[][] readIntMat(int row, int col)
	{
		int[][] mat = new int[row][col];
		String line;
		Scanner scan = null;
		int i = 0, j = 0;
		try {
			while (i < row && (line = reader.readLine()) != null)
			{
				if (line.trim().length() == 0)	continue;
				scan = new Scanner(line);
				scan.useDelimiter(delimiter);
				while (j < col && scan.hasNextInt())
					mat[i][j++] = scan.nextInt();
				j = 0;
				++ i;
			}
			if (scan == null)	return null; // no lines left
			scan.close();
			return mat;
		}
		catch (Exception e) {
			e.printStackTrace(); return null;
		}
	}

	/**
	 * Read the whole CSV file. No need to specify rows/columns. <br>
	 * The size of the first line determines the matrix width
	 * @return an int matrix from CSV file
	 */
	public int[][] readIntMat()
	{
		ArrayList<String[]> rowBuffer = new ArrayList<>(512);
		String line;
		try {
			while ((line = reader.readLine()) != null)
			{
				line = line.trim();
				if (line.length() == 0)	continue;
				rowBuffer.add(line.split(delimiter));
			}

			int iLen = rowBuffer.size();
			if (iLen == 0)	return null;
			int jLen = rowBuffer.get(0).length;
			int[][] mat = new int[iLen][jLen];

			for (int i = 0; i < iLen; i++)
				for (int j = 0; j < jLen; j++)
				{
					String[] toks = rowBuffer.get(i);
					mat[i][j] = (j < toks.length && toks[j].length() > 0) ? Integer.parseInt(toks[j]) : 0;
				}

			return mat;
		}
		catch (IOException e) {
			e.printStackTrace(); return null;
		}
	}


	/**
	 * Capable of resuming reading
	 * @return an int array (single-row-matrix)
	 */
	public int[] readIntVec(int col)
	{
		int[] array = new int[col];
		try {
			int i = 0;
			while (i < col)
			{
				if (colScanner == null || !colScanner.hasNextInt())
				{
					String line = reader.readLine();
					if (line == null)
					{
						colScanner = null;
						return i == 0 ? null : array; // if nothing read, return null
					}
					colScanner = new Scanner(line);
					colScanner.useDelimiter(delimiter);
				}
				while (i < col && colScanner.hasNextInt())
					array[i++] = colScanner.nextInt();
			}

			// not filled up yet, we proceed to the next row if there is any
			return array;
		}
		catch (IOException e) {
			e.printStackTrace(); return null;
		}
	}

	/**
	 * Read a one CSV file.
	 * @param all true to read the whole CSV as one array. False to read one line at a time
	 * @return an int array from CSV file
	 */
	public int[] readIntVec(boolean all)
	{
		ArrayList<String[]> rowBuffer = new ArrayList<>(64);
		String line;
		try {
			int count = 0;
			while ((line = reader.readLine()) != null)
			{
				line = line.trim();
				if (line.length() == 0)	continue;
				String[] toks = line.split(delimiter);
				count += toks.length;
				rowBuffer.add(toks);
				if (!all) break; // read only one line at a time and return
			}
			if (rowBuffer.size() == 0)	return null;

			int[] array = new int[count];

			int pt = 0;
			for (int i = 0; i < rowBuffer.size(); i++)
				for (String s : rowBuffer.get(i))
					array[pt++] = s.length() > 0 ? Integer.parseInt(s) : 0;

					return array;
		}
		catch (IOException e) {
			e.printStackTrace(); return null;
		}
	}

	/**
	 * Default to false: read only one line at a time
	 */
	public int[] readIntVec()	{	return readIntVec(false);	}


	/**
	 * Capable of resuming read
	 * @param row read a specified number of rows
	 * @param col read a specified number of columns
	 * @return a float matrix from CSV file
	 */
	public float[][] readFloatMat(int row, int col)
	{
		float[][] mat = new float[row][col];
		String line;
		Scanner scan = null;
		int i = 0, j = 0;
		try {
			while (i < row && (line = reader.readLine()) != null)
			{
				if (line.trim().length() == 0)	continue;
				scan = new Scanner(line);
				scan.useDelimiter(delimiter);
				while (j < col && scan.hasNextFloat())
					mat[i][j++] = scan.nextFloat();
				j = 0;
				++ i;
			}
			if (scan == null)	return null; // no lines left
			scan.close();
			return mat;
		}
		catch (IOException e) {
			e.printStackTrace(); return null;
		}
	}

	/**
	 * Read the whole CSV file. No need to specify rows/columns. <br>
	 * The size of the first line determines the matrix width
	 * @return an float matrix from CSV file
	 */
	public float[][] readFloatMat()
	{
		ArrayList<String[]> rowBuffer = new ArrayList<>(512);
		String line;
		try {
			while ((line = reader.readLine()) != null)
			{
				line = line.trim();
				if (line.length() == 0)	continue;
				rowBuffer.add(line.split(delimiter));
			}

			int iLen = rowBuffer.size();
			if (iLen == 0)	return null;
			int jLen = rowBuffer.get(0).length;
			float[][] mat = new float[iLen][jLen];

			for (int i = 0; i < iLen; i++)
				for (int j = 0; j < jLen; j++)
				{
					String[] toks = rowBuffer.get(i);
					mat[i][j] = (j < toks.length && toks[j].length() > 0) ? Float.parseFloat(toks[j]) : 0;
				}

			return mat;
		}
		catch (IOException e) {
			e.printStackTrace(); return null;
		}
	}

	/**
	 * Capable of resuming reading
	 * @return a float array (single-row-matrix)
	 */
	public float[] readFloatVec(int col)
	{
		float[] array = new float[col];
		try {
			int i = 0;
			while (i < col)
			{
				if (colScanner == null || !colScanner.hasNextFloat())
				{
					String line = reader.readLine();
					if (line == null)
					{
						colScanner = null;
						return i == 0 ? null : array; // if nothing read, return null
					}
					colScanner = new Scanner(line);
					colScanner.useDelimiter(delimiter);
				}
				while (i < col && colScanner.hasNextFloat())
					array[i++] = colScanner.nextFloat();
			}

			// not filled up yet, we proceed to the next row if there is any
			return array;
		}
		catch (IOException e) {
			e.printStackTrace(); return null;
		}
	}

	/**
	 * Read a one CSV file.
	 * @param all true to read the whole CSV as one array. False to read one line at a time
	 * @return an float array from CSV file
	 */
	public float[] readFloatVec(boolean all)
	{
		ArrayList<String[]> rowBuffer = new ArrayList<>(64);
		String line;
		try {
			int count = 0;
			while ((line = reader.readLine()) != null)
			{
				line = line.trim();
				if (line.length() == 0)	continue;
				String[] toks = line.split(delimiter);
				count += toks.length;
				rowBuffer.add(toks);
				if (!all) break; // read only one line at a time and return
			}
			if (rowBuffer.size() == 0)	return null;

			float[] array = new float[count];

			int pt = 0;
			for (int i = 0; i < rowBuffer.size(); i++)
				for (String s : rowBuffer.get(i))
					array[pt++] = s.length() > 0 ? Float.parseFloat(s) : 0;

					return array;
		}
		catch (IOException e) {
			e.printStackTrace(); return null;
		}
	}

	/**
	 * Default to false: read only one line at a time
	 */
	public float[] readFloatVec()	{	return readFloatVec(false);	}


	/**
	 * Capable of resuming read
	 * @param row read a specified number of rows
	 * @param col read a specified number of columns
	 * @return a string matrix from CSV file
	 */
	public String[][] readStringMat(int row, int col)
	{
		String[][] mat = new String[row][col];
		String line;
		Scanner scan = null;
		int i = 0, j = 0;
		try {
			while (i < row && (line = reader.readLine()) != null)
			{
				if (line.trim().length() == 0)	continue;
				scan = new Scanner(line);
				scan.useDelimiter(delimiter);
				while (j < col && scan.hasNext())
					mat[i][j++] = scan.next();
				j = 0;
				++ i;
			}
			if (scan == null)	return null; // no lines left
			scan.close();
			return mat;
		}
		catch (IOException e) {
			e.printStackTrace(); return null;
		}
	}


	/**
	 * Read the whole CSV file. No need to specify rows/columns. <br>
	 * The size of the first line determines the matrix width
	 * @return a string matrix from CSV file
	 */
	public String[][] readStringMat()
	{
		ArrayList<String[]> rowBuffer = new ArrayList<>(512);
		String line;
		try {
			while ((line = reader.readLine()) != null)
			{
				line = line.trim();
				if (line.length() == 0)	continue;
				rowBuffer.add(line.split(delimiter));
			}

			int iLen = rowBuffer.size();
			if (iLen == 0)	return null;
			int jLen = rowBuffer.get(0).length;
			String[][] mat = new String[iLen][jLen];

			for (int i = 0; i < iLen; i++)
				for (int j = 0; j < jLen; j++)
				{
					String[] toks = rowBuffer.get(i);
					mat[i][j] = j < toks.length ? toks[j] : null;
				}

			return mat;
		}
		catch (IOException e) {
			e.printStackTrace(); return null;
		}
	}

	/**
	 * Capable of resuming reading
	 * @return a string array (single-row-matrix)
	 */
	public String[] readStringVec(int col)
	{
		String[] array = new String[col];
		try {
			int i = 0;
			while (i < col)
			{
				if (colScanner == null || !colScanner.hasNext())
				{
					String line = reader.readLine();
					if (line == null)
					{
						colScanner = null;
						return i == 0 ? null : array; // if nothing read, return null
					}
					colScanner = new Scanner(line);
					colScanner.useDelimiter(delimiter);
				}
				while (i < col && colScanner.hasNext())
					array[i++] = colScanner.next();
			}

			// not filled up yet, we proceed to the next row if there is any
			return array;
		}
		catch (IOException e) {
			e.printStackTrace(); return null;
		}
	}

	/**
	 * Read a one CSV file.
	 * @param all true to read the whole CSV as one array. False to read one line at a time
	 * @return a String array from CSV file
	 */
	public String[] readStringVec(boolean all)
	{
		ArrayList<String[]> rowBuffer = new ArrayList<>(64);
		String line;
		try {
			int count = 0;
			while ((line = reader.readLine()) != null)
			{
				line = line.trim();
				if (line.length() == 0)	continue;
				String[] toks = line.split(delimiter);
				count += toks.length;
				rowBuffer.add(toks);
				if (!all) break; // read only one line at a time and return
			}
			if (rowBuffer.size() == 0)	return null;

			String[] array = new String[count];

			int pt = 0;
			for (int i = 0; i < rowBuffer.size(); i++)
				for (String s : rowBuffer.get(i))
					array[pt++] = s;

			return array;
		}
		catch (IOException e) {
			e.printStackTrace(); return null;
		}
	}

	/**
	 * Default to false: read only one line at a time
	 */
	public String[] readStringVec()	{	return readStringVec(false);	}


	/**
	 * Load an entire int matrix from a file
	 * @param fileName extension can be either "txt" (CSV file) or "dat" to load from binary java object file
	 */
	public static int[][] readIntMat(String fileName)
	{
		if (fileExtension(fileName).equals("dat"))
		{
			return new Pickle<int[][]>().load(fileName);
		}
		else // "txt"
		{
			CsvReader csv = new CsvReader(fileName);
			return csv.readIntMat();
		}
	}


	/**
	 * Load an entire int array from a file
	 * @param fileName extension can be either "txt" (CSV file) or "dat" to load from binary java object file
	 */
	public static int[] readIntVec(String fileName)
	{
		if (fileExtension(fileName).equals("dat"))
		{
			return new Pickle<int[]>().load(fileName);
		}
		else // "txt"
		{
			CsvReader csv = new CsvReader(fileName);
			return csv.readIntVec(true); // read all lines as one array
		}
	}

	/**
	 * Load an entire float matrix from a file
	 * @param fileName extension can be either "txt" (CSV file) or "dat" to load from binary java object file
	 */
	public static float[][] readFloatMat(String fileName)
	{
		if (fileExtension(fileName).equals("dat"))
		{
			return new Pickle<float[][]>().load(fileName);
		}
		else // "txt"
		{
			CsvReader csv = new CsvReader(fileName);
			return csv.readFloatMat();
		}
	}

	/**
	 * Load an entire float array from a file
	 * @param fileName extension can be either "txt" (CSV file) or "dat" to load from binary java object file
	 */
	public static float[] readFloatVec(String fileName)
	{
		if (fileExtension(fileName).equals("dat"))
		{
			return new Pickle<float[]>().load(fileName);
		}
		else // "txt"
		{
			CsvReader csv = new CsvReader(fileName);
			return csv.readFloatVec(true); // read all lines as one array
		}
	}

	/**
	 * Load an entire string matrix from a file
	 * @param fileName extension can be either "txt" (CSV file) or "dat" to load from binary java object file
	 */
	public static String[][] readStringMat(String fileName)
	{
		if (fileExtension(fileName).equals("dat"))
		{
			return new Pickle<String[][]>().load(fileName);
		}
		else // "txt"
		{
			CsvReader csv = new CsvReader(fileName);
			return csv.readStringMat();
		}
	}

	/**
	 * Load an entire string array from a file
	 * @param fileName extension can be either "txt" (CSV file) or "dat" to load from binary java object file
	 */
	public static String[] readStringVec(String fileName)
	{
		if (fileExtension(fileName).equals("dat"))
		{
			return new Pickle<String[]>().load(fileName);
		}
		else // "txt"
		{
			CsvReader csv = new CsvReader(fileName);
			return csv.readStringVec(true); // read all lines as one array
		}
	}

	private static String fileExtension(String fileName)
	{
		// Extract the file extension
		String ext = "";

		int dot = fileName.lastIndexOf('.');
		int f = Math.max(fileName.lastIndexOf('/'), fileName.lastIndexOf('\\'));

		if (dot > f)
		{
			ext = fileName.substring(dot+1);
		}
		return ext;
	}
}
