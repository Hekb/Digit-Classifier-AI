#include "SimpleFeedForwardNetwork.h"
#include <iostream>
#include <random>
#include <iomanip>      // std::setprecision

void SimpleFeedForwardNetwork::initialize(int seed)
{
	srand(seed);
	hiddenLayerWeights.resize(inputLayerSize);

	for (size_t i = 0; i < inputLayerSize; i++)
	{
		hiddenLayerWeights[i].resize(hiddenLayerSize);
		for (size_t j = 0; j < hiddenLayerSize; j++)
		{
			hiddenLayerWeights[i][j] = (rand() % 100 + 1) * 1.0 / 100; 	// This network cannot learn if the initial weights are set to zero.
		}
	}
	outputLayerWeights.resize(hiddenLayerSize);
	for (size_t i = 0; i < hiddenLayerSize; i++)
	{
		outputLayerWeights[i] = (rand() % 100 + 1) * 1.0 / 100; 	// This network cannot learn if the initial weights are set to zero.
	}
}

void SimpleFeedForwardNetwork::train(const vector< vector< double > >& x,
	const vector< double >& y, size_t numEpochs)
{
	size_t trainingexamples = x.size();


	// train the network
	for (size_t epoch = 0; epoch < numEpochs; epoch++)
	{
		// print
		cout << "epoch = " << epoch << ", outputs =";
		for (size_t example = 0; example < trainingexamples; example++)
		{
			// propagate the inputs forward to compute the outputs 
			vector< double > activationInput(inputLayerSize); // We store the activation of each node (over all input and hidden layers) as we need that data during back propagation.			
			for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) // initialize input layer with training data
			{
				activationInput[inputNode] = x[example][inputNode];
			}

			vector< double > activationHidden(hiddenLayerSize);
			// calculate activations of hidden layers (for now, just one hidden layer)
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			{
				double inputToHidden = 0;
				for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++)
				{
					inputToHidden += hiddenLayerWeights[inputNode][hiddenNode] * activationInput[inputNode];
				}
				activationHidden[hiddenNode] = g(inputToHidden);
			}

			// one output node.
			double inputAtOutput = 0;
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++){
				inputAtOutput += outputLayerWeights[hiddenNode] * activationHidden[hiddenNode];
			}
			
			double activationOutput = g(inputAtOutput);
			cout << " " << std::setprecision(2) << activationOutput;
			


			
			// calculating errors
			double errorOfOutputNode = gprime(activationOutput) * (y[example] - activationOutput);

			// Calculating error of hidden layer. Special calculation since we only have one output node; i.e. no summation over next layer nodes
			// Also adjusting weights of output layer
			vector< double > errorOfHiddenNode(hiddenLayerSize);
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			{
				errorOfHiddenNode[hiddenNode] = outputLayerWeights[hiddenNode] * errorOfOutputNode;
				errorOfHiddenNode[hiddenNode] *= gprime(activationHidden[hiddenNode]);
			}

			//adjusting weights
			//adjusting weights at output layer
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			{
				outputLayerWeights[hiddenNode] += alpha * activationHidden[hiddenNode] * errorOfOutputNode;
			}

			// Adjusting weights at hidden layer.
			for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++)
			{
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					hiddenLayerWeights[inputNode][hiddenNode] += alpha * activationInput[inputNode] * errorOfHiddenNode[hiddenNode];
				}
			}
		}
		cout << endl;
	}

	return;
}
