import torch
import math
def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient (float)
		- direction: Unit vector (torch.Tensor) in direction of steepest ascent
		- descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
	"""
	# Your code here
	magnitude = 0
	for i in range(len(gradient)):
		magnitude += gradient[i]**2
	magnitude = math.sqrt(magnitude)

	if magnitude == 0:
		direction = torch.zeros(len(gradient))
	else:	
		direction = torch.tensor([gradient[i]/magnitude for i in range(len(gradient))])

	result = {}
	result["magnitude"] = magnitude
	result["direction"] = direction
	result["descent_direction"] = -1*direction

	return result