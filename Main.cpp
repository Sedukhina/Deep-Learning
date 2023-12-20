#pragma once

#include <utility>
#include <random>

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>

#include "Neuron.h"
#include "Perceptron.h"
#include "InputFunctions.h"
#include "ActivationFunctions.h"
#include "CPNN.h"

inline void PrintFloatVector(std::vector<float> vec);
inline bool IsNumber(const std::string& s);
const std::pair < std::vector<std::vector<float>>, std::vector<float>> ReadKDD(std::string path);
std::pair < std::vector<std::vector<float>>, std::vector<float>> GenerateWeightsSumFunc(unsigned int InputLayerSize, unsigned int Amount, float Min, float Max, bool Round = false);


int main() 
{
	printf("Probabilistic neural network\n");
	std::pair<std::vector<std::vector<float>>, std::vector<float>> Input = ReadKDD("Data//KDD_train.csv");
	CPNN PNN = CPNN(Input);

	int Right = 0;
	int Wrong = 0;

	std::pair<std::vector<std::vector<float>>, std::vector<float>> TestData = ReadKDD("Data//KDD_test.csv");
	for (size_t i = 0; i < TestData.first.size(); i++)
	{
		if (PNN.Recognise(TestData.first[i]) == TestData.second[i])
		{
			Right++;
		}
		else
		{
			Wrong++;
		}
	}
	printf("Right: %i\nWrong: %i\n", Right, Wrong);
}

void PrintFloatVector(std::vector<float> vec)
{
	for (size_t i = 0; i < vec.size(); i++)
	{
		printf("%f; ", vec[i]);
	}
}

const std::pair<std::vector<std::vector<float>>, std::vector<float>> ReadKDD(std::string path)
{
	std::string filename{ path };
	std::ifstream input{ filename };

	if (!input.is_open()) {
		std::cerr << "Couldn't read file: " << filename << "\n";
		return { {}, {} };
	}

	//std::vector<std::vector<float>, float> csvRows;
	std::pair < std::vector<std::vector<float>>, std::vector<float>> Data;

	for (std::string line; std::getline(input, line);) 
	{
		std::istringstream ss(std::move(line));
		std::vector<float> row;
		if (!Data.first.empty()) {
			// We expect each row to be as big as the first row
			row.reserve(Data.first.front().size());
		}
		if (line.find("teardrop") != std::string::npos)
		{
			Data.second.push_back(1);
			printf("1");
		}
		else
		{
			Data.second.push_back(0);
		}
		// std::getline can split on other characters, here we use ','
		for (std::string value; std::getline(ss, value, ',');) {
			if (IsNumber(value))
				row.push_back(std::stof(value));
		}
		Data.first.push_back(std::move(row));
	}
	printf("\n");
	return Data;
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

bool IsNumber(const std::string& s)
{
	std::string::const_iterator it = s.begin();
	while (it != s.end() && std::isdigit(*it)) ++it;
	return !s.empty() && it == s.end();
}