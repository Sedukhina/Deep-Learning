#include "ActivationFunctions.h"
#include <cmath>

float ActivationFunctions::LogisticSigmoidFunction(float X)
{
	return 1 / (1 + exp(-X));
}