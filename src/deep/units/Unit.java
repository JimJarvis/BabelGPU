package deep.units;

import deep.LearningPlan;

public abstract class Unit
{
	public String name;
	protected LearningPlan learningPlan;
	protected boolean debug = false;
	
	public Unit(String name)
	{
		this.name = name;
	}
	
	public void enableDebug(boolean debug)
	{
		this.debug = debug;
	}
	
	public void setLearningPlan(LearningPlan learningPlan)
	{
		this.learningPlan = learningPlan;
	}
}
