package utils;

/**
 * Time the code
 * @author Jim Fan  (c) 2014
 */
public class Timer
{
	private long timest;
	private long timel;

	/**
	 * Singleton pattern
	 */
	private static final Timer instance = new Timer();
	private Timer() { start();	}

	public static Timer getInstance() {	return instance;	}

	/**
	 * Restart the timer
	 */
	public void start() { timel = timest = System.currentTimeMillis(); }

	/**
	 * Read from the timer start()
	 */
	public void readFromStart()
	{
		long timel = System.currentTimeMillis();
		System.out.format("[Timer] %.2f sec\n", ((double)(timel - timest))/1000);
	}

	/**
	 * Read from last read
	 */
	public void readFromLast()
	{
		long now = System.currentTimeMillis();
		System.out.format("[Timer] %.2f sec\n", ((double)(now - timel))/1000);
		timel = now;
	}
}
