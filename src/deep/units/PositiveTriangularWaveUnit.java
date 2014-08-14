package deep.units;

import gpu.*;

public class PositiveTriangularWaveUnit extends ElementComputeUnit
{
	private static final long serialVersionUID = 1L;

	public PositiveTriangularWaveUnit(String name, InletUnit inlet, boolean hasBias, float scalor)
	{
		super(name, inlet, hasBias, scalor);
	}

	public PositiveTriangularWaveUnit(String name, InletUnit inlet, float scalor)
	{
		super(name, inlet, scalor);
	}
	
	public PositiveTriangularWaveUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	@Override
	public void forward_element(float scalor)
	{
		Thrust.trianglar_wave_positive(input.data(), output.data(), (float) (Math.PI/2), scalor);
	}

	@Override
	public void backward_element()
	{
		throw new UnsupportedOperationException(
				"Triangular wave unit doesn't yet support backprop");
	}

}
