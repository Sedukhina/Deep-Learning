#include "InputFunctions.h"

float InputFunctions::WeightedInputSum(std::vector<float> Input, std::vector<float> Weights)
{
	float Sum = 0;

	for (int i = 0; i < Input.size(); i++)
	{
		Sum += Input[i] * Weights[i];
	}

	return Sum;
}

float InputFunctions::MaxWeightedInput(std::vector<float> Input, std::vector<float> Weights)
{
	float Max = -std::numeric_limits<double>::infinity();
	for (int i = 0; i < Input.size(); i++)
	{	
		if (Max < Input[i] * Weights[i]) {
				Max = Input[i] * Weights[i];
		}
	}
	return Max;
}

float InputFunctions::WeightedInputMultiplication(std::vector<float> Input, std::vector<float> Weights)
{
	float MultiplicationResult = 1;
	for (int i = 0; i < Input.size(); i++)
	{
		MultiplicationResult *= Input[i] * Weights[i];
	}
	return MultiplicationResult;
}

float InputFunctions::MinWeightedInput(std::vector<float> Input, std::vector<float> Weights)
{
	float Min = std::numeric_limits<double>::infinity();
	for (int i = 0; i < Input.size(); i++)
	{
		if (Min > Input[i] * Weights[i]) {
			Min = Input[i] * Weights[i];
		}
	}
	return Min;
}