#include "Neuron.h"
#include <random>

float CNeuron::Recognition(const std::vector<float>& Input, const std::function<float(std::vector<float>, std::vector<float>)>& InputFunction, const std::function<float(float)>& ActivationFunction) const
{
	if (Input.size() + (int)!Elementary != InputWeights.size())
	{
		printf("CNeuron::Recognition error: Input vector size is not the same as InputWeights size\n");
		return -1;
	}
	std::vector<float>Inp = Input;
	if (!Elementary)
	{
		//If Neuron has connection with itself it always has input 1 there 
		Inp.insert(Inp.begin(), 1.f);
	}
	//Using Input Function
	float InputResult = InputFunction(Inp, InputWeights);
	//Returning Result of Activation function
	return ActivationFunction(InputResult);
}

void CNeuron::Learn(const std::vector<float> &Input, float RealResult, int CyclesToLearn, const std::function<float(std::vector<float>, std::vector<float>)> &InputFunction, const std::function<float(float)> &ActivationFunction)
{
	if (Input.size()+(int)!Elementary != InputWeights.size())
	{
		printf("CNeuron::Learn error: Input vector size is not the same as InputWeights size\n");
		return;
	}
	for (int CurrentCycle = 0; CurrentCycle < CyclesToLearn; CurrentCycle++)
	{
		float RecognitionResult = Recognition(Input, InputFunction, ActivationFunction);
		printf("Recognition result: %f\n", RecognitionResult);
		WeightsCorrection(RecognitionResult, Input, RealResult);
	}
}

float CNeuron::WeightsCorrection(float RecognitionResult, const std::vector<float>& Input, float RealResult, bool HiddenLayer)
{
	float Delta;
	if (HiddenLayer)
	{
		Delta = RecognitionResult * (1 - RecognitionResult) * (RealResult);
	}
	else
	{
		Delta = RecognitionResult * (1 - RecognitionResult) * (RealResult - RecognitionResult);
	}
	if (!Elementary)
	{
		InputWeights[0] += Delta;
	}
	//Correcting Coefficients
	for (int i = 0; i < Input.size(); i++)
	{
		InputWeights[i+(int)!Elementary] += Delta * Input[i];
	}
	return Delta;
}

void CNeuron::GenerateInitialWights(unsigned int InputsAmount)
{
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_real_distribution<> Dist(0.05, 0.955);
	printf("Generated weights: \n");
	for (int i = 0; i < InputsAmount+(int)!Elementary; i++)
	{
		InputWeights.push_back((float)Dist(gen));
		printf("%f\n", InputWeights[i]);
	}
}
