package utils;
import java.io.*;

/**
 * Simulate python "pickle" with Serializable
 * @author Jim Fan  (c) 2014
 */
public class Pickle<T extends Serializable>
{
	
	/**
	 * Dump an object to a specified file
	 */
	public void dump(T obj, String fileName)
	{
		try
		{
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName));
			oos.writeObject(obj);
			oos.close();
		}
		catch (IOException e)
		{	System.err.println("File not found: " + fileName);	}
	}

	/**
	 * Load an object from a specified file
	 */
	public T load(String fileName)
	{
		try
		{
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fileName));
			T obj = (T) ois.readObject();
			ois.close();
			return obj;
		}
		catch (IOException e)
		{
			System.err.println("File not found: " + fileName);
			return null;
		}
		catch (ClassNotFoundException e)
		{
			System.err.println("File not found: " + fileName);
			return null;
		}
	}
}
