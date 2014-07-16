package deep;

public class LearningPlan
{
	public float lr; // learning rate
	public float reg; // regularization
	public float gamma; 
	public int totalTrainSize;
	public int totalEpochs;
	public int curTrainSize = 0;
	public int curEpoch = 0;
	
	public LearningPlan() {};
	
	public LearningPlan(float lr, float reg, float gamma, int totalTrainSize, int totalEpochs)
	{
		this.lr = lr;
		this.reg = reg;
		this.gamma = gamma;
		this.totalTrainSize = totalTrainSize;
		this.totalEpochs = totalEpochs;
	}
	
	/**
	 * Prepare for re-run
	 */
	public void reset() { this.curTrainSize = 0; }
	
	/**
	 * Do we use regularization?
	 */
	public boolean hasReg()
	{
		return this.reg > 0;
	}
}
