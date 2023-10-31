#include "CPNN.h"
#include <cmath>
#include <unordered_map>

float CPNN::Recognise(const std::vector<float>& Input)
{
	std::unordered_map<float, float> AdditionLayerOutput;
	for (auto el : Classes)
	{
		AdditionLayerOutput[el] = 0.f;
	}
	//For each neuron
	for (size_t i = 0; i < PatternLayer.first.size(); i++)
	{
		float SqSum = 0;
		for (size_t j = 0; j < PatternLayer.first[i].size(); j++)
		{
			SqSum += (PatternLayer.first[i][j] - Input[j]) * (PatternLayer.first[i][j] - Input[j]);
		}
		AdditionLayerOutput[PatternLayer.second[i]] = AdditionLayerOutput[PatternLayer.second[i]] + std::exp(-SqSum / SigmaSq);
	}

	auto Max = std::max_element(std::begin(AdditionLayerOutput), std::end(AdditionLayerOutput),
		[](const std::pair<float, float>& p1, const std::pair<float, float>& p2) 
		{return p1.second < p2.second;});

	return Max->first;
}
