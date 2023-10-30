#pragma once
#include <vector>
#include <functional>

class CNeuron;

class CPerceptron
{
public:
	CPerceptron() = delete;
	explicit CPerceptron(int LayersAmount, bool ElementaryNeuron = false);
	explicit CPerceptron(std::vector<int> PerceptronStruct, bool ElementaryNeuron = false);
	explicit CPerceptron(const std::vector<std::vector<CNeuron>>& NeuronsVec) : Neurons(NeuronsVec) {};

	void Learn(const std::vector<std::vector<float>>& Input, std::vector<float> RealResult, int ErasToLearn, const std::function<float(std::vector<float>, std::vector<float>)>& InputFunction, const std::function<float(float)>& ActivationFunction);
	std::vector<float>  Recognize(const std::vector<float>& Input, const std::function<float(std::vector<float>, std::vector<float>)>& InputFunction, const std::function<float(float)>& ActivationFunction) const;

	void PrintPerceptronStructure() const;
	void PrintPerceptronWeights() const;

private:
	//Functions for perceptron creation
	void GenerateLayer(unsigned int AmountOfNeuronOnLayer, unsigned int AmountOfNeuronOnPrevLayer);
	void CreateInputLayer(unsigned int InputsAmount);

	std::vector<std::vector<CNeuron>> Neurons;
	bool ElementaryNeurons;
};

