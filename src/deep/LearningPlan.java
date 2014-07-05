package deep;

public class LearningPlan
{
	public float lr; // learning rate
	public float reg; // regularization
	public float gamma; 
	public int trainingN;
	
	public LearningPlan(float lr, float reg, float gamma, int trainingN)
	{
		this.lr = lr;
		this.reg = reg;
		this.gamma = gamma;
		this.trainingN = trainingN;
	}
}
