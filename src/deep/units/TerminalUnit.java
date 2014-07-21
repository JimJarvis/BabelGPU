package deep.units;

import gpu.*;

import java.util.ArrayList;
import java.util.Collections;

import deep.DeepException;

public abstract class TerminalUnit extends ComputeUnit
{
	/**
	 * Pure loss function
	 */
	protected float lossPure = 0;
	/**
	 * Loss due to regularization
	 */
	protected float lossReg = 0;
	
	/**
	 * During training, we might want to turn off loss calculation
	 */
	public boolean doesCalcLoss = true;
	
	public TerminalUnit(String name, InletUnit inlet, boolean hasBias)
	{
		super(name, inlet, 1, hasBias);
	}
	
	public TerminalUnit(String name, InletUnit inlet)
	{
		this(name, inlet,  true);
	}
	
	@Override
	public void setup()
	{
		setupLink();
	}
	
	/**
	 * Subclasses should implement 'forward_terminal'
	 * 'final' keyword could have prevented subclass from overriding forward()
	 */
	@Override
	public void forward()
	{
		inlet.nextGold();
		
		// Update regularization loss
		this.lossReg += learningPlan.regScheme.regularize();
		
		if (hasBias)
			input.data().fillLastRow0();

		// Will be implemented by subclasses
		this.lossPure += forward_terminal(this.doesCalcLoss);
	}

	/**
	 * Ensure that you call super.forward() first
	 * @param doesCalcLoss if false, we don't explicitly calculate the loss
	 * @return total pure loss to be updated (should NOT be normalized by batchSize)
	 */
	protected abstract float forward_terminal(boolean doesCalcLoss);
	
	/**
	 * If there's a summation, then the backward gradient should be dividied by batchSize
	 */
	protected float batchNormalizer() {	return 1f / inlet.batchSize;	}
	
	public float lossTotal() { return lossPure() + lossReg() ;	}
	
	/**
	 * Normalized by curTrainSize (number of training samples seen so far)
	 */
	public float lossPure()
	{
		if (!doesCalcLoss)
			throw new DeepException("Loss is not being calculated");
		return this.lossPure / learningPlan.doneSampleSize;
	}
	
	/**
	 * Loss due to regularization
	 */
	public float lossReg()
	{	
		if (!doesCalcLoss)
			throw new DeepException("Loss is not being calculated");
		return this.lossReg;
	}

	/**
	 * Reset both lossPure and lossReg
	 */
	public void clearLoss()
	{ 
		this.lossPure = 0;
		this.lossReg = 0;
	}
	
	/**
	 * During training, we might not need to calculate the loss explicitly
	 * @param do we want to calculate it? Default = true
	 */
	public void setCalcLoss(boolean doesCalcLoss)
	{
		this.doesCalcLoss = doesCalcLoss;
	}
}
