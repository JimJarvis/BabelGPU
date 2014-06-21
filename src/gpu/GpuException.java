package gpu;

/**
 * Unchecked exception
 */
@SuppressWarnings("serial")
public class GpuException extends RuntimeException
{
	public GpuException()
	{
		super();
	}
	
	public GpuException(String msg)
	{
		super(msg);
	}
}
