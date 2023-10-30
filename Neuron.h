#pragma once
#include <memory>
#include <functional>
#include <vector>

class CNeuron
{
public:
	CNeuron() = delete;
	explicit CNeuron(const std::vector<float>& vecInputsWeights, bool Elementary = false) : InputWeights(vecInputsWeights), Elementary(Elementary) { };
	explicit CNeuron(unsigned int iInputsAmount, bool Elementary = false) : Elementary(Elementary) { GenerateInitialWights(iInputsAmount); };

	//Function to train single neuron
	void Learn(const std::vector<float> &Input, float RealResult, int CyclesToLearn, const std::function<float(std::vector<float>, std::vector<float>)> &InputFunction, const std::function<float(float)> &ActivationFunction);
	float Recognition(const std::vector<float>& Input, const std::function<float(std::vector<float>, std::vector<float>)>& InputFunction, const std::function<float(float)>& ActivationFunction) const;
	//Function to correct weights used when training neuron separetly or in a network; Returns delta which can be needed for perceptron learning
	float WeightsCorrection(float RecognitionResult, const std::vector<float>& Input, float RealResult, bool HiddenLayer = false);

	std::vector<float> GetInputWeights() const { return InputWeights; };

private:
	void GenerateInitialWights(unsigned int iInputsAmount);

	//Weights for each input
	std::vector<float> InputWeights;
	bool Elementary;
};

