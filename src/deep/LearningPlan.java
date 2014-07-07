package deep;

public class LearningPlan
{
	public float lr; // learning rate
	public float reg; // regularization
	public float gamma; 
	public int totalTrainSize;
	public int curTrainSize = 0;
	
	public LearningPlan() {};
	
	public LearningPlan(float lr, float reg, float gamma, int totalTrainSize)
	{
		this.lr = lr;
		this.reg = reg;
		this.gamma = gamma;
		this.totalTrainSize = totalTrainSize;
	}
	
	/**
	 * Prepare for re-run
	 */
	public void reset() { this.curTrainSize = 0; }
}
