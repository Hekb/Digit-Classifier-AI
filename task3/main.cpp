#include <vector>
#include <iostream>
#include <string>
#include "MNIST_reader.h"
#include "MNIST_reader.h"
#include "SimpleFeedForwardNetwork.h"

using namespace std;
int main(){
	
	//load training MNIST images
	string filename = "../MNIST/train-images.idx3-ubyte";
	vector <vector< int> > training_images;
	loadMnistImages(filename, training_images);
	//Scale to 0-1
	vector <vector< double> > x (training_images.size());
	for(int i = 0; i < training_images.size(); i++){
		x[i].resize(training_images[i].size());
		for(int j = 0; j < training_images[i].size(); j++){
			x[i][j] = (double)training_images[i][j] / 255;
		}
	}

	//load training MNIST labels	
	filename = "../MNIST/train-labels.idx1-ubyte";
	vector<int> training_labels;
	loadMnistLabels(filename, training_labels);
	
	//load test MNIST images	
	filename = "../MNIST/t10k-images.idx3-ubyte";
	vector <vector< int> > test_images;
	loadMnistImages(filename, test_images );
	vector <vector< double> > testImages (test_images.size());
	//Scale 0-1
	for(int i = 0; i < test_images.size(); i++){
		testImages[i].resize(test_images[i].size());
		for(int j = 0; j < test_images[i].size(); j++){
			testImages[i][j] = (double)test_images[i][j] / 255;
		}
	}

	//load test MNIST labels
	filename = "../MNIST/t10k-labels.idx1-ubyte";
	vector<int> testLabels;
	loadMnistLabels(filename, testLabels);

	//Divide the training set into a training set and validation set
	vector<vector<double>> training_set (4000);
	vector<double> training_set_labels (4000);
	vector<vector<double>> validation_set (2000);
	vector<double> validation_set_labels (2000);
	for(int i = 0; i < 4000; i++){
		training_set[i] = x[i];
	}
	for(int i = 0; i < 2000; i++){
		validation_set[i] = x[i + 4000];
	}
	for(int i = 0; i < 4000; i++){
		training_set_labels[i] = training_labels[i];
	}
	for(int i = 0; i < 2000; i++){
		validation_set_labels[i] = training_labels[i + 4000];
	}

	//hyper-parameters for part 1
	double alpha = 0.2;   // learning rate
	size_t inputLayerSize = 784;
	size_t hiddenLayerSize = 32;
	size_t layerSize = 4;
	size_t outputLayerSize = 10;

	long seed = 2133451174; // random seed for the part 1


	SimpleFeedForwardNetwork nn(alpha, hiddenLayerSize, inputLayerSize, layerSize, outputLayerSize);
	nn.initialize(seed);
	cout << "Running part 1 with ten output nodes" << endl;
	cout << "------------------------------------" << endl;
	//Train the network with 10 output nodes
	nn.train(training_set, training_set_labels, validation_set, validation_set_labels, 15);
	vector<int> testingOutput (10000);
	testingOutput = nn.testData(testImages);
	int rightGuesses = 0;
	int incorrectGuesses = 0;
	for(int i = 0; i < 10000; i++){
		if(testingOutput[i] == testLabels[i]){
			rightGuesses++;
		}else{
			incorrectGuesses++;
		}
	}
	cout << endl;
	cout << "Correct guesses: " << rightGuesses << endl;
	cout << "Incorrect guesses: " << incorrectGuesses << endl;
	cout << "Test guesses accuracy: " << ((double)rightGuesses/10000)*100 << "%" << endl;

	//hyper-parameters for part 2
	alpha = 0.1;   // learning rate
	inputLayerSize = 784;
	hiddenLayerSize = 32;
	layerSize = 4;
	outputLayerSize = 4;
	cout << endl;
	cout << endl;
	cout << "Running part 2 with four output nodes" << endl;
	cout << "------------------------------------" << endl;
	seed = 2993330707; // random seed for the part 2

	SimpleFeedForwardNetwork nn2(alpha, hiddenLayerSize, inputLayerSize, layerSize, outputLayerSize);
	nn2.initialize(seed);
	//Train the data
	nn2.binaryEncodingTrain(training_set, training_set_labels, validation_set, validation_set_labels, 23);

	//Test the data and then compare with the actual outputs
	testingOutput = nn2.testData2(testImages);

	rightGuesses = 0;
	incorrectGuesses = 0;
	for(int i = 0; i < 10000; i++){
		if(testingOutput[i] == testLabels[i]){
			rightGuesses++;
		}else{
			incorrectGuesses++;
		}
	}
	cout << endl;
	cout << "Correct guesses: " << rightGuesses << endl;
	cout << "Incorrect guesses: " << incorrectGuesses << endl;
	cout << "Test guesses accuracy: " << ((double)rightGuesses/10000)*100 << "%" << endl;
	

	return 0;
}