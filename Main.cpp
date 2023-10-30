#pragma once

#include <utility>
#include <random>

#include "Neuron.h"
#include "Perceptron.h"
#include "InputFunctions.h"
#include "ActivationFunctions.h"

inline void PrintFloatVector(std::vector<float> vec);
std::pair < std::vector<std::vector<float>>, std::vector<float>> GenerateLearningWeightsSumFunc(unsigned int InputLayerSize, unsigned int Amount);

int main() 
{
	printf("1.1 Classical Neuron\n");
	//Creating case to train on
	std::vector<float> Input{0.2f, 0.3f, 0.7f};
	float InputResult = 0.9f;
	printf("Learning Case: ");
	PrintFloatVector(Input);
	printf("-> %f\n", InputResult);

	//Creating Neuron - using constructor which generates initial weights
	CNeuron TestNeuron = CNeuron(3);

	//Initializing Lerning Process
	TestNeuron.Learn(Input, InputResult, 20, &InputFunctions::WeightedInputSum, &ActivationFunctions::LogisticSigmoidFunction);
	printf("Weights After Learning\n");
	for (auto weight : TestNeuron.GetInputWeights())
	{
		printf("%f\n", weight);
	}

	printf("\n\n\n1.2 Elementary perceptrone 1-1-1\n");
	//Creating case to train on
	std::vector<float> InputElemPerc{ 0.6f };
	float InputElemPercResult = 0.9f;
	printf("Learning Case: ");
	PrintFloatVector(InputElemPerc);
	printf("-> %f\n", InputElemPercResult);

	//Creating perceptron - waking up constructor by the structure 
	std::vector<int> ElementaryPerceptronStruct{1, 1, 1};
	CPerceptron ElementaryPerceptrone = CPerceptron(ElementaryPerceptronStruct, true);

	//Outputing weights for each neuron 
	ElementaryPerceptrone.PrintPerceptronWeights();
	printf("Result of recognition with initial weights\n");
	std::vector<float> ElementaryPerceptronResult = ElementaryPerceptrone.Recognize({ InputElemPerc }, &InputFunctions::WeightedInputSum, &ActivationFunctions::LogisticSigmoidFunction);
	PrintFloatVector(ElementaryPerceptronResult);
	printf("\n");

	ElementaryPerceptrone.Learn({ {InputElemPerc} }, { InputElemPercResult }, 20, &InputFunctions::WeightedInputSum, &ActivationFunctions::LogisticSigmoidFunction);

	ElementaryPerceptrone.PrintPerceptronWeights();
	printf("Result of recognition after learning\n");
	ElementaryPerceptronResult = ElementaryPerceptrone.Recognize({ 0.6f }, &InputFunctions::WeightedInputSum, &ActivationFunctions::LogisticSigmoidFunction);
	PrintFloatVector(ElementaryPerceptronResult);
	printf("\n");

	printf("\n\n\n1.3 Perceptrone with 2-3-1 structure\n");
	//Creating case to train on
	std::vector<float> InputPerc{ 0.6f, 0.3f };
	float InputPercResult = 0.9f;
	printf("Test Case: ");
	PrintFloatVector(InputPerc);
	printf("-> %f\n", InputPercResult);

	//Creating perceptron - waking up constructor by the structure 
	std::vector<int> PerceptronStruct{ 2, 3, 1 };
	CPerceptron Perceptrone = CPerceptron(PerceptronStruct);
	Perceptrone.PrintPerceptronWeights();
	printf("Result of test case recognition with initial weights\n");
	std::vector<float> PerceptronResult = Perceptrone.Recognize({ InputPerc }, &InputFunctions::WeightedInputSum, &ActivationFunctions::LogisticSigmoidFunction);
	PrintFloatVector(PerceptronResult);
	printf("\n");

	std::pair<std::vector<std::vector<float>>, std::vector<float>> InputToLearnPerc = GenerateLearningWeightsSumFunc(2, 1000);

	for (size_t i = 0; i < InputToLearnPerc.first.size(); i++)
	{
		printf("Learning Case %i: ", i);
		PrintFloatVector(InputToLearnPerc.first[i]);
		printf("-> %f\n", InputToLearnPerc.second[i]);
	}
	Perceptrone.Learn({ {InputToLearnPerc.first} }, { InputToLearnPerc.second }, 20, &InputFunctions::WeightedInputSum, &ActivationFunctions::LogisticSigmoidFunction);

	Perceptrone.PrintPerceptronWeights();
	printf("Result of recognition after learning\n");
	PerceptronResult = Perceptrone.Recognize({ InputPerc }, &InputFunctions::WeightedInputSum, &ActivationFunctions::LogisticSigmoidFunction);
	PrintFloatVector(PerceptronResult);
	printf("\n");
}

void PrintFloatVector(std::vector<float> vec)
{
	for (size_t i = 0; i < vec.size(); i++)
	{
		printf("%f; ", vec[i]);
	}
}

std::pair<std::vector<std::vector<float>>, std::vector<float>> GenerateLearningWeightsSumFunc(unsigned int InputAmount, unsigned int Amount)
{
	std::vector<std::vector<float>> Inputs;
	std::vector<float > Results;

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_real_distribution<> Dist(0.05, 0.955);

	for (size_t i = 0; i < Amount; i++)
	{
		Inputs.push_back({ (float)Dist(gen), (float)Dist(gen) });
		Results.push_back(Inputs[i][0] + Inputs[i][1]);
	}
	return std::make_pair(Inputs, Results);
}
