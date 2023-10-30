#pragma once

#include <vector>
#include <limits>
#include <map>

namespace InputFunctions 
{
	float WeightedInputSum(std::vector<float> Input, std::vector<float> Weights);

	float MaxWeightedInput(std::vector<float> Input, std::vector<float> Weights);

	float WeightedInputMultiplication(std::vector<float> Input, std::vector<float> Weights);

	float MinWeightedInput(std::vector<float> Input, std::vector<float> Weights);
}