#pragma once
#include <vector>
#include <set>

//Probabilistic neural network
class CPNN
{
public:
	//Pair of vectors: first one is each cases weights, second is each cases result
	explicit CPNN(const std::pair < std::vector<std::vector<float>>, std::vector<float>>& Data) : PatternLayer(Data), Classes(PatternLayer.second.begin(), PatternLayer.second.end()){ };
	explicit CPNN(const std::pair < std::vector<std::vector<float>>, std::vector<float>>& Data, const std::set<float> &ClassSet) : PatternLayer(Data), Classes(ClassSet) {};
	float Recognise(const std::vector<float>& Input);

private:
	std::pair < std::vector<std::vector<float>>, std::vector<float>> PatternLayer;
	std::set<float> Classes;

	const float SigmaSq = 0.1f;
};

