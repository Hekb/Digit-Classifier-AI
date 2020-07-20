#pragma once
#include <math.h>
#include <vector>

using namespace std;

class SimpleFeedForwardNetwork
{
public:
	void initialize(int seed);

	void train(const vector< vector< double > >& x,
		const vector< double >& y, size_t numEpochs);

	SimpleFeedForwardNetwork(double alpha, size_t hiddenLayerSize, size_t inputLayerSize) :
		alpha(alpha), hiddenLayerSize(hiddenLayerSize), inputLayerSize(inputLayerSize) {}

private:
	vector< vector< double > > hiddenLayerWeights; // [from][to]
	vector< double > outputLayerWeights;

	double alpha;
	size_t hiddenLayerSize;
	size_t inputLayerSize;


	inline double g(double x) {return 1.0 / (1.0 + exp(-x)); }
	inline double gprime(double y) {return y * (1 - y); }
};
