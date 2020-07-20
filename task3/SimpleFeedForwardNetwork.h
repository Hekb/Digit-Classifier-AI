#pragma once
#include <math.h>
#include <vector>

using namespace std;

class SimpleFeedForwardNetwork {
public:
  void initialize(int seed);

  void train(const vector<vector<double>> &x, const vector<double> &y,
             const vector<vector<double>> &validation_set,
             const vector<double> &validation_labels, size_t numEpochs);

  SimpleFeedForwardNetwork(double alpha, size_t hiddenLayerSize,
                           size_t inputLayerSize, size_t layerSize,
                           size_t outputLayerSize)
      : alpha(alpha), hiddenLayerSize(hiddenLayerSize),
        inputLayerSize(inputLayerSize), layerSize(layerSize),
        outputLayerSize(outputLayerSize) {}
  int predication(vector<double> activationOutput);
  vector<int> testData(vector<vector<double>> testData);

  void binaryEncodingTrain(const vector<vector<double>> &traintingset,
                           const vector<double> &trainingsetlabels,
                           const vector<vector<double>> &validation_set,
                           const vector<double> &validation_labels,
                           size_t numEpochs);

  vector<int> testData2(vector<vector<double>> testData);

private:
  vector<vector<vector<double>>> hiddenLayerWeights; // [from][to]
  vector<vector<double>> outputLayerWeights;

  double alpha;
  size_t hiddenLayerSize;
  size_t inputLayerSize;
  size_t layerSize;
  size_t outputLayerSize;
  /*
          inline double g(double x) {
                  return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
          }
          inline double gprime(double y) {return 1 - (tanh(y)*tanh(y)); }
          */

  inline double g(double x) { return 1.0 / (1.0 + exp(-x)); }
  inline double gprime(double y) { return y * (1 - y); }
};
