#include "Perceptron.h"
#include "Neuron.h"
#include <random>

CPerceptron::CPerceptron(int LayersAmount, bool ElementaryNeuron) : ElementaryNeurons(ElementaryNeuron)
{
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<> Dist(1, 10);

	std::vector<int> AmountOfNeuronOnLayer;
	printf("Generating layers: \n");

	int AmountOnPrevLayer = Dist(gen);
	CreateInputLayer(AmountOnPrevLayer);
	
	//Creating other layers
	int AmountOnLayer = 0;
	for (int i = 1; i < LayersAmount; i++)
	{
		AmountOnLayer = (Dist(gen));
		GenerateLayer(AmountOnLayer, AmountOnPrevLayer);
		AmountOnPrevLayer = AmountOnLayer;
	}
}

CPerceptron::CPerceptron(std::vector<int> PerceptronStruct, bool ElementaryNeuron) : ElementaryNeurons(ElementaryNeuron)
{
	CreateInputLayer(PerceptronStruct[0]);

	for (int i = 1; i < PerceptronStruct.size(); i++)
	{
		GenerateLayer(PerceptronStruct[i], PerceptronStruct[i-1]);
	}
}

void CPerceptron::Learn(const std::vector<std::vector<float>>& Input, std::vector<float> RealResult, int ErasToLearn, const std::function<float(std::vector<float>, std::vector<float>)>& InputFunction, const std::function<float(float)>& ActivationFunction)
{
	for (int CurrentEra = 0; CurrentEra < ErasToLearn; CurrentEra++)
	{
		for (int InputIndex = 0; InputIndex < Input.size(); InputIndex++) 
		{
			//Caching Output for weight correction 
			std::vector<std::vector<float>> Output{ Input[InputIndex] };
			std::vector<float> CurrentInput = Input[InputIndex];
			for (int LayerNum = 1; LayerNum < Neurons.size(); LayerNum++)
			{
				Output.push_back({});
				for (int NeuronNum = 0; NeuronNum < Neurons[LayerNum].size(); NeuronNum++)
				{
					Output[LayerNum].push_back(Neurons[LayerNum][NeuronNum].Recognition(CurrentInput, InputFunction, ActivationFunction));
				}
				CurrentInput.clear();
				CurrentInput.assign(Output[LayerNum].begin(), Output[LayerNum].end());
			}

			//Saving weights to use it in the next step PreviousWights[Layer][Neuron][Weight]
			std::vector<std::vector<std::vector<float>>> PreviousWeights;
			for (int LayerNum = 0; LayerNum < Neurons.size(); LayerNum++)
			{
				PreviousWeights.push_back({});
				for (int NeuronNum = 0; NeuronNum < Neurons[LayerNum].size(); NeuronNum++)
				{
					PreviousWeights[LayerNum].push_back({});
					PreviousWeights[LayerNum][NeuronNum] = Neurons[LayerNum][NeuronNum].GetInputWeights();
				}
			}
			//Weight Correction starts
			for (int NeuronNum = 0; NeuronNum < Neurons[Neurons.size() - 1].size(); NeuronNum++)
			{
				//Saving correction delta in output vector(on a result place)
				Output[Neurons.size() - 1][NeuronNum] = Neurons[Neurons.size() - 1][NeuronNum].WeightsCorrection(Output[Neurons.size() - 1][NeuronNum], Output[Neurons.size() - 2], RealResult[InputIndex]);
			}
			for (int LayerNum = Neurons.size()-2; LayerNum > 0; LayerNum--)
			{
				for (int NeuronNum = 0; NeuronNum < Neurons[LayerNum].size(); NeuronNum++)
				{
					std::vector<float> PreviousOutputWeightsForNeuron = {};
					for (int NeuronN = 0; NeuronN < Neurons[LayerNum+1].size(); NeuronN++)
					{
						PreviousOutputWeightsForNeuron.push_back(PreviousWeights[LayerNum+1][NeuronN][NeuronNum+(int)!ElementaryNeurons]);
					}
					Output[LayerNum][NeuronNum] = Neurons[LayerNum][NeuronNum].WeightsCorrection(Output[LayerNum][NeuronNum], Output[LayerNum-1], InputFunction(Output[LayerNum + 1], PreviousOutputWeightsForNeuron), true);
				}
			}
		}
	}
}

std::vector<float> CPerceptron::Recognize(const std::vector<float>& Input, const std::function<float(std::vector<float>, std::vector<float>)>& InputFunction, const std::function<float(float)>& ActivationFunction) const
{
	//Output from previous layer is input to the next one 
	std::vector<float> CurrentInput = Input;
	std::vector<float> NextLayerInput;
	for (int LayerNum = 1; LayerNum < Neurons.size(); LayerNum++)
	{
		//For each Neuron on layer calling recognition function
		for (int NeuronNum = 0; NeuronNum < Neurons[LayerNum].size(); NeuronNum++)
		{
			NextLayerInput.push_back(Neurons[LayerNum][NeuronNum].Recognition(CurrentInput, InputFunction, ActivationFunction));
		}
		//Assigning recognition result vector as input for the next layer 
		CurrentInput.clear();
		CurrentInput.assign(NextLayerInput.begin(), NextLayerInput.end());
		//Clearing vector to record next layer results
		NextLayerInput.clear();
	}
	//Returning last layer output
	return CurrentInput;
}

void CPerceptron::PrintPerceptronStructure() const
{
	printf("\nPerceprtron Structure:\n");
	for (auto Layer : Neurons)
	{
		printf("Layer: ");
		for (auto Neuron : Layer)
		{
			printf(" o ");
		}
		printf("\n");
	}
	printf("\n\n");
}

void CPerceptron::PrintPerceptronWeights() const
{
	printf("\nPerceprtron Weights:\n");
	for(int LayerNum = 0; LayerNum < Neurons.size(); LayerNum++)
	{
		printf("Layer %i:\n", LayerNum);
		for (int NeuronNum = 0; NeuronNum < Neurons[LayerNum].size(); NeuronNum++)
		{
			printf("	Neuron %i:\n", NeuronNum+1);
			auto Weights = Neurons[LayerNum][NeuronNum].GetInputWeights();
			for (int Weight = 0; Weight < Weights.size(); Weight++)
			{
				printf("		Weight %i: %f\n", Weight, Weights[Weight]);
			}
		}
		printf("\n");
	}
	printf("\n\n");
}

void CPerceptron::GenerateLayer(unsigned int AmountOfNeuronOnLayer, unsigned int AmountOfNeuronsOnPrevLayer)
{
	std::vector<CNeuron> NeuronVec;
	for (int i = 0; i < AmountOfNeuronOnLayer; i++)
	{
		NeuronVec.push_back(CNeuron(AmountOfNeuronsOnPrevLayer, ElementaryNeurons));
	}
	Neurons.push_back(NeuronVec);
}

void CPerceptron::CreateInputLayer(unsigned int InputsAmount)
{
	//First layer is input layer designed only to take an input, weight is stabely 1 and not corrected during learning process
	std::vector<CNeuron> Entrance;
	std::vector<float> InputLayerWeight{ 1 };
	for (int i = 0; i < InputsAmount; i++)
	{
		Entrance.push_back(CNeuron(InputLayerWeight));
	}
	//Pushing Input layer to a neuron vector
	Neurons.push_back(Entrance);
}
