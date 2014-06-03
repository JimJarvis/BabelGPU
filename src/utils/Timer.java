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
	 * Optional event argument
	 */
	public void readFromStart(String event)
	{
		long timel = System.currentTimeMillis();
		timeFormat(event, timel - timest);
	}
	
	public void readFromStart() {	readFromStart("Timer"); }

	/**
	 * Read from last read
	 * Optional event argument
	 */
	public void readFromLast(String event)
	{
		long now = System.currentTimeMillis();
		timeFormat(event, now - timel);
		timel = now;
	}
	
	public void readFromLast() {	readFromLast("Timer");	}
	
	/**
	 * Set timing precision
	 * default: 2 decimal places
	 */
	private static int prec = 2;
	public static void setPrecision(int prec) {	Timer.prec = prec; }
	
	// helper
	private void timeFormat(String event, long delta)
	{
		System.out.format(
				"[%s] %." + Timer.prec + "f sec\n", event, ((double)delta)/1000);
	}
}
