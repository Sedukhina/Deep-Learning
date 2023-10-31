#pragma once

#include <utility>
#include <random>

#include "Neuron.h"
#include "Perceptron.h"
#include "InputFunctions.h"
#include "ActivationFunctions.h"
#include "CPNN.h"

inline void PrintFloatVector(std::vector<float> vec);
std::pair < std::vector<std::vector<float>>, std::vector<float>> GenerateWeightsSumFunc(unsigned int InputLayerSize, unsigned int Amount, float Min, float Max, bool Round = false);

int main() 
{
	printf("Probabilistic neural network\n");
	std::pair<std::vector<std::vector<float>>, std::vector<float>> Input = GenerateWeightsSumFunc(2, 1000, 20, 120, true);

	for (size_t i = 0; i < Input.first.size(); i++)
	{
		printf("Learning Case %i: ", i);
		PrintFloatVector(Input.first[i]);
		printf("-> %f\n", Input.second[i]);
	}

	CPNN PNN = CPNN(Input);


	printf("\n\nTesting Cases in same range\n");

	std::pair<std::vector<std::vector<float>>, std::vector<float>> TestCases1 = GenerateWeightsSumFunc(2, 10, 20, 120, true);

	for (size_t i = 0; i < TestCases1.first.size(); i++)
	{
		printf("Case %i: ", i);
		PrintFloatVector(TestCases1.first[i]);
		printf("-> %f\n", TestCases1.second[i]);

		float RecResult = PNN.Recognise(TestCases1.first[i]);
		printf("Recognition result = %f\n", RecResult);
		printf("Differnce: %f\n\n", std::abs(TestCases1.second[i]-RecResult));
	}

	printf("\n\nTesting Cases in wider range\n");

	std::pair<std::vector<std::vector<float>>, std::vector<float>> TestCases2 = GenerateWeightsSumFunc(2, 10, 0, 200, true);

	for (size_t i = 0; i < TestCases2.first.size(); i++)
	{
		printf("Case %i: ", i);
		PrintFloatVector(TestCases2.first[i]);
		printf("-> %f\n", TestCases2.second[i]);

		float RecResult = PNN.Recognise(TestCases2.first[i]);
		printf("Recognition result = %f\n", RecResult);
		printf("Differnce: %f\n\n", std::abs(TestCases1.second[i] - RecResult));
	}
}

void PrintFloatVector(std::vector<float> vec)
{
	for (size_t i = 0; i < vec.size(); i++)
	{
		printf("%f; ", vec[i]);
	}
}

std::pair<std::vector<std::vector<float>>, std::vector<float>> GenerateWeightsSumFunc(unsigned int InputAmount, unsigned int Amount, float Min, float Max, bool Round)
{
	std::vector<std::vector<float>> Inputs;
	std::vector<float > Results;

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_real_distribution<> Dist(Min, Max);

	for (size_t i = 0; i < Amount; i++)
	{
		Inputs.push_back({});
		float Sum = 0;
		for (size_t j = 0; j < InputAmount; j++) 
		{
			float NewWeight = Dist(gen);
			if (Round)
			{
				NewWeight = std::round(NewWeight);
			}
			Inputs[i].push_back(NewWeight);
			Sum += NewWeight;
		}
		Results.push_back(Sum);
	}
	return std::make_pair(Inputs, Results);
}
