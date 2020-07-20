#include "SimpleFeedForwardNetwork.h"
#include <iomanip> // std::setprecision
#include <iostream>
#include <random>

// Neural Network
void SimpleFeedForwardNetwork::initialize(int seed) {
  srand(seed);
  // Create the neural network with x hidden layers
  hiddenLayerWeights.resize(layerSize);
  int weightsAmount = (inputLayerSize * hiddenLayerSize) +
                      ((hiddenLayerSize * hiddenLayerSize) * (layerSize - 1)) +
                      (hiddenLayerSize * outputLayerSize);

  // Random uniform intiaization
  const double range_from = -0.5;
  const double range_to = 0.5;
  random_device rand_dev;
  mt19937 generator(seed);
  uniform_real_distribution<double> distr(range_from, range_to);
  vector<double> wg(weightsAmount);
  for (int i = 0; i < weightsAmount; ++i) {
    wg[i] = distr(generator);
  }
  int ctr = 0;

  for (size_t i = 0; i < layerSize; i++) {
    if (i == 0) {
      hiddenLayerWeights[i].resize(inputLayerSize);
    } else {
      hiddenLayerWeights[i].resize(hiddenLayerSize);
    }
    for (size_t j = 0; j < hiddenLayerWeights[i].size(); j++) {
      hiddenLayerWeights[i][j].resize(hiddenLayerSize);
      for (size_t x = 0; x < hiddenLayerWeights[i][j].size(); x++) {
        hiddenLayerWeights[i][j][x] = wg[ctr];
        ctr++;
      }
    }
  }
  outputLayerWeights.resize(hiddenLayerSize);
  for (size_t i = 0; i < hiddenLayerSize; i++) {
    outputLayerWeights[i].resize(outputLayerSize);
    for (size_t j = 0; j < outputLayerSize; j++) {
      outputLayerWeights[i][j] = wg[ctr];
      ctr++;
    }
  }
}
// prediction used for part 1
int SimpleFeedForwardNetwork::predication(vector<double> activationOutput) {
  int index = 0;
  double max = -0.000000004;
  for (size_t i = 0; i < 10; i++) {
    if (activationOutput[i] > max) {
      max = activationOutput[i];
      index = i;
    }
  }
  return index;
}

// train the network for part 1
void SimpleFeedForwardNetwork::train(
    const vector<vector<double>> &traintingset,
    const vector<double> &trainingsetlabels,
    const vector<vector<double>> &validation_set,
    const vector<double> &validation_labels, size_t numEpochs) {

  size_t trainingexamples = traintingset.size();
  size_t validationexamples = validation_set.size();

  // train the network
  for (size_t epoch = 0; epoch < numEpochs; epoch++) {
    double training_error = 0;
    int correctTrainGuesses = 0;

    cout << "epoch = " << epoch << endl;
    for (size_t example = 0; example < trainingexamples; example++) {

      // propagate the inputs forward to compute the outputs
      vector<double> activationInput(inputLayerSize);
      // We store the activation of each node (over all
      // input and hidden layers) as we need that data
      // during back propagation.
      // initialize input layer with training data
      for (size_t i = 0; i < inputLayerSize; i++) {
        activationInput[i] = traintingset[example][i];
      }

      vector<vector<double>> activationHidden(layerSize);
      // calculate activations of hidden layers
      for (size_t layer = 1; layer <= layerSize; layer++) {
        // Loops for every node at layer i
        activationHidden[layer - 1].resize(hiddenLayerSize);
        for (size_t j = 0; j < hiddenLayerSize; j++) {
          // We are the first later => look at the input activation
          double inputToHid = 0;
          if (layer == 1) {
            for (size_t i = 0; i < inputLayerSize; i++) {
              inputToHid +=
                  hiddenLayerWeights[layer - 1][i][j] * activationInput[i];
            }
          } else {
            for (size_t i = 0; i < hiddenLayerSize; i++) {
              inputToHid += hiddenLayerWeights[layer - 1][i][j] *
                            activationHidden[layer - 2][i];
            }
          }
          activationHidden[layer - 1][j] = g(inputToHid);
        }
      }

      // activation output
      vector<double> activationOutput(outputLayerSize);
      double temp = 0;
      for (size_t j = 0; j < outputLayerSize; j++) {
        for (size_t x = 0; x < hiddenLayerSize; x++) {
          temp += outputLayerWeights[x][j] *
                  activationHidden[activationHidden.size() - 1][x];
        }
        activationOutput[j] = g(temp);
        temp = 0;
      }

      double tt = 0;
      int index = trainingsetlabels[example];
      vector<double> actualOutput(10);
      actualOutput[index] = 1;
      // now calculate the error loss
      for (size_t j = 0; j < outputLayerSize; j++) {
        tt += (activationOutput[j] - actualOutput[j]) *
              (activationOutput[j] - actualOutput[j]);
      }
      training_error += tt;
      // calculate the accuracy
      if (trainingsetlabels[example] == predication(activationOutput)) {
        correctTrainGuesses++;
      }

      vector<double> errorOfOutputLayer(outputLayerSize);
      // Now we calc output error
      for (size_t j = 0; j < outputLayerSize; j++) {
        errorOfOutputLayer[j] = gprime(activationOutput[j]) *
                                (actualOutput[j] - activationOutput[j]);
      }

      // Now update weights at output layer
      for (int from = 0; from < hiddenLayerSize; from++) {
        for (int to = 0; to < outputLayerSize; to++) {
          outputLayerWeights[from][to] +=
              alpha * activationHidden[layerSize - 1][from] *
              errorOfOutputLayer[to];
        }
      }

      // Now calc error of hidden layers
      vector<vector<double>> errorOfHiddenNode(layerSize);
      for (int l = (layerSize - 1); l >= 0; l--) {
        errorOfHiddenNode[l].resize(hiddenLayerSize);

        for (size_t node = 0; node < hiddenLayerSize; node++) {

          double temp = 0;
          if (l == (layerSize - 1)) {
            // last layer
            for (size_t x = 0; x < outputLayerSize; x++) {
              temp += outputLayerWeights[node][x] * errorOfOutputLayer[x];
            }
            errorOfHiddenNode[l][node] =
                temp * gprime(activationHidden[l][node]);
          } else {
            for (size_t x = 0; x < hiddenLayerSize; x++) {
              temp += hiddenLayerWeights[l + 1][node][x] *
                      errorOfHiddenNode[l + 1][x];
            }
            errorOfHiddenNode[l][node] =
                temp * gprime(activationHidden[l][node]);
          }
        }
      }
      // Adjust weights for hidden layers
      for (int i = layerSize - 1; i >= 0; i--) {
        for (int j = 0; j < hiddenLayerWeights[i].size(); j++) {
          for (int z = 0; z < hiddenLayerSize; z++) {
            if (i == 0) {
              hiddenLayerWeights[i][j][z] +=
                  alpha * activationInput[j] * errorOfHiddenNode[i][z];
            } else {
              hiddenLayerWeights[i][j][z] +=
                  alpha * activationHidden[i - 1][j] * errorOfHiddenNode[i][z];
            }
          }
        }
      }
    }

    // Now run the validation set without adjusting the weights
    double validation_error = 0;
    double correctValidation = 0;
    for (size_t valexample = 0; valexample < validationexamples; valexample++) {

      // propagate the inputs forward to compute the outputs
      vector<double> activationInput(
          inputLayerSize); // We store the activation of each node (over all
                           // input and hidden layers) as we need that data
                           // during back propagation.
      // initialize input layer with training data
      for (size_t i = 0; i < inputLayerSize; i++) {
        activationInput[i] = validation_set[valexample][i];
      }

      // calculate activations of hidden layers
      vector<vector<double>> activationHidden(layerSize);
      for (size_t layer = 1; layer <= layerSize; layer++) {
        // Loops for every node at layer i
        activationHidden[layer - 1].resize(hiddenLayerSize);
        for (size_t j = 0; j < hiddenLayerSize; j++) {
          // We are the first later => look at the input activation
          double inputToHid = 0;
          if (layer == 1) {
            for (size_t i = 0; i < inputLayerSize; i++) {
              inputToHid +=
                  hiddenLayerWeights[layer - 1][i][j] * activationInput[i];
            }
          } else {
            for (size_t i = 0; i < hiddenLayerSize; i++) {
              inputToHid += hiddenLayerWeights[layer - 1][i][j] *
                            activationHidden[layer - 2][i];
            }
          }
          activationHidden[layer - 1][j] = g(inputToHid);
        }
      }

      // activation output
      vector<double> activationOutput(outputLayerSize);
      double temp = 0;
      for (size_t j = 0; j < outputLayerSize; j++) {
        for (size_t x = 0; x < hiddenLayerSize; x++) {
          temp += outputLayerWeights[x][j] *
                  activationHidden[activationHidden.size() - 1][x];
        }
        activationOutput[j] = g(temp);
        temp = 0;
      }
      // Calculate error loss
      double tt = 0;
      int index = validation_labels[valexample];
      vector<double> actualOutput(10);
      actualOutput[index] = 1;
      for (size_t j = 0; j < outputLayerSize; j++) {
        tt += (activationOutput[j] - actualOutput[j]) *
              (activationOutput[j] - actualOutput[j]);
      }
      validation_error += tt;
      // Calculate the accuracy
      if (validation_labels[valexample] == predication(activationOutput)) {
        correctValidation++;
      }
    }

    // Print the data
    cout << "Training loss: "
         << ((double)training_error / trainingexamples) * 100 << "%" << endl;
    cout << "Training accuracy: "
         << ((double)correctTrainGuesses / trainingexamples) * 100 << "%"
         << endl;

    cout << "Validation loss: "
         << ((double)validation_error / validationexamples) * 100 << "%"
         << endl;
    cout << "Validation accuracy: "
         << ((double)correctValidation / validationexamples) * 100 << "%"
         << endl;
  }

  return;
}

// Used to test the data for part 1
vector<int> SimpleFeedForwardNetwork::testData(vector<vector<double>> data) {
  vector<int> output(data.size());
  for (size_t valexample = 0; valexample < data.size(); valexample++) {
    // propagate the inputs forward to compute the outputs
    vector<double> activationInput(
        inputLayerSize); // We store the activation of each node (over all input
                         // and hidden layers) as we need that data during back
                         // propagation.
    // initialize input layer with training data
    for (size_t i = 0; i < inputLayerSize; i++) {
      activationInput[i] = data[valexample][i];
    }

    vector<vector<double>> activationHidden(layerSize);
    // calculate activations of hidden layers
    for (size_t layer = 1; layer <= layerSize; layer++) {
      // Loops for every node at layer i
      activationHidden[layer - 1].resize(hiddenLayerSize);
      for (size_t j = 0; j < hiddenLayerSize; j++) {
        // We are the first later => look at the input activation
        double inputToHid = 0;
        if (layer == 1) {
          for (size_t i = 0; i < inputLayerSize; i++) {
            inputToHid +=
                hiddenLayerWeights[layer - 1][i][j] * activationInput[i];
          }
        } else {
          for (size_t i = 0; i < hiddenLayerSize; i++) {
            inputToHid += hiddenLayerWeights[layer - 1][i][j] *
                          activationHidden[layer - 2][i];
          }
        }
        activationHidden[layer - 1][j] = g(inputToHid);
      }
    }

    // activation output
    vector<double> activationOutput(outputLayerSize);
    double temp = 0;
    for (size_t j = 0; j < outputLayerSize; j++) {
      for (size_t x = 0; x < hiddenLayerSize; x++) {
        temp += outputLayerWeights[x][j] *
                activationHidden[activationHidden.size() - 1][x];
      }
      activationOutput[j] = g(temp);
      temp = 0;
    }

    output[valexample] = predication(activationOutput);
  }
  return output;
}
// Used to test the data for part 2
vector<int> SimpleFeedForwardNetwork::testData2(vector<vector<double>> data) {
  vector<int> output(data.size());
  for (size_t valexample = 0; valexample < data.size(); valexample++) {
    // propagate the inputs forward to compute the outputs
    vector<double> activationInput(
        inputLayerSize); // We store the activation of each node (over all input
                         // and hidden layers) as we need that data during back
                         // propagation.
    // initialize input layer with training data
    for (size_t i = 0; i < inputLayerSize; i++) {
      activationInput[i] = data[valexample][i];
    }

    vector<vector<double>> activationHidden(layerSize);
    // calculate activations of hidden layers
    for (size_t layer = 1; layer <= layerSize; layer++) {
      // Loops for every node at layer i
      activationHidden[layer - 1].resize(hiddenLayerSize);
      for (size_t j = 0; j < hiddenLayerSize; j++) {
        // We are the first later => look at the input activation
        double inputToHid = 0;
        if (layer == 1) {
          for (size_t i = 0; i < inputLayerSize; i++) {
            inputToHid +=
                hiddenLayerWeights[layer - 1][i][j] * activationInput[i];
          }
        } else {
          for (size_t i = 0; i < hiddenLayerSize; i++) {
            inputToHid += hiddenLayerWeights[layer - 1][i][j] *
                          activationHidden[layer - 2][i];
          }
        }
        activationHidden[layer - 1][j] = g(inputToHid);
      }
    }

    // activation output
    vector<double> activationOutput(outputLayerSize);
    double temp = 0;
    for (size_t j = 0; j < outputLayerSize; j++) {
      for (size_t x = 0; x < hiddenLayerSize; x++) {
        temp += outputLayerWeights[x][j] *
                activationHidden[activationHidden.size() - 1][x];
      }
      activationOutput[j] = g(temp);
      temp = 0;
    }

    // Convert the output to a number
    double activationOutputGuess = 0;
    for (int i = 0; i < 4; i++) {
      if (i == 0) {
        if (activationOutput[i] >= 0.9) {
          activationOutputGuess += 1;
        }
      } else if (i == 1) {
        if (activationOutput[i] >= 0.9) {
          activationOutputGuess += 2;
        }
      } else if (i == 2) {
        if (activationOutput[i] >= 0.9) {
          activationOutputGuess += 4;
        }
      } else {
        if (activationOutput[i] >= 0.9) {
          activationOutputGuess += 8;
        }
      }
    }

    output[valexample] = activationOutputGuess;
  }
  return output;
}

// Used to train the network for part 2
void SimpleFeedForwardNetwork::binaryEncodingTrain(
    const vector<vector<double>> &traintingset,
    const vector<double> &trainingsetlabels,
    const vector<vector<double>> &validation_set,
    const vector<double> &validation_labels, size_t numEpochs) {

  size_t trainingexamples = traintingset.size();
  size_t validationexamples = validation_set.size();

  // train the network
  for (size_t epoch = 0; epoch < numEpochs; epoch++) {
    double training_error = 0;
    int correctTrainGuesses = 0;

    cout << "epoch = " << epoch << endl;
    for (size_t example = 0; example < trainingexamples; example++) {

      // propagate the inputs forward to compute the outputs
      vector<double> activationInput(
          inputLayerSize); // We store the activation of each node (over all
                           // input and hidden layers) as we need that data
                           // during back propagation.
      // initialize input layer with training data
      for (size_t i = 0; i < inputLayerSize; i++) {
        activationInput[i] = traintingset[example][i];
      }

      vector<vector<double>> activationHidden(layerSize);
      // calculate activations of hidden layers
      for (size_t layer = 1; layer <= layerSize; layer++) {
        // Loops for every node at layer i
        activationHidden[layer - 1].resize(hiddenLayerSize);
        for (size_t j = 0; j < hiddenLayerSize; j++) {
          // We are the first later => look at the input activation
          double inputToHid = 0;
          if (layer == 1) {
            for (size_t i = 0; i < inputLayerSize; i++) {
              inputToHid +=
                  hiddenLayerWeights[layer - 1][i][j] * activationInput[i];
            }
          } else {
            for (size_t i = 0; i < hiddenLayerSize; i++) {
              inputToHid += hiddenLayerWeights[layer - 1][i][j] *
                            activationHidden[layer - 2][i];
            }
          }
          activationHidden[layer - 1][j] = g(inputToHid);
        }
      }

      // activation output
      vector<double> activationOutput(outputLayerSize);
      double temp = 0;
      for (size_t j = 0; j < outputLayerSize; j++) {
        for (size_t x = 0; x < hiddenLayerSize; x++) {
          temp += outputLayerWeights[x][j] *
                  activationHidden[activationHidden.size() - 1][x];
        }
        activationOutput[j] = g(temp);
        temp = 0;
      }

      // convert to binary
      vector<int> actualOutput(4);
      int number = trainingsetlabels[example];
      for (int i = 0; number > 0; i++) {
        actualOutput[i] = number % 2;
        number = number / 2;
      }
      // Calcualte the error loss
      double tt = 0;
      for (size_t j = 0; j < outputLayerSize; j++) {
        tt += (activationOutput[j] - actualOutput[j]) *
              (activationOutput[j] - actualOutput[j]);
      }
      training_error += tt;
      // Convert the output to a number
      double activationOutputGuess = 0;

      for (int i = 0; i < 4; i++) {
        if (i == 0) {
          if (activationOutput[i] >= 0.9) {
            activationOutputGuess += 1;
          }
        } else if (i == 1) {
          if (activationOutput[i] >= 0.9) {
            activationOutputGuess += 2;
          }
        } else if (i == 2) {
          if (activationOutput[i] >= 0.9) {
            activationOutputGuess += 4;
          }
        } else {
          if (activationOutput[i] >= 0.9) {
            activationOutputGuess += 8;
          }
        }
      }
      // Calcualate accuracy
      if (trainingsetlabels[example] == activationOutputGuess) {
        correctTrainGuesses++;
      }

      vector<double> errorOfOutputLayer(outputLayerSize);
      // Now we calc output error
      for (size_t j = 0; j < outputLayerSize; j++) {
        errorOfOutputLayer[j] = gprime(activationOutput[j]) *
                                (actualOutput[j] - activationOutput[j]);
      }

      // Now update weights at output layer
      for (int from = 0; from < hiddenLayerSize; from++) {
        for (int to = 0; to < outputLayerSize; to++) {
          outputLayerWeights[from][to] +=
              alpha * activationHidden[layerSize - 1][from] *
              errorOfOutputLayer[to];
        }
      }

      // Now calc error of hidden layers
      vector<vector<double>> errorOfHiddenNode(layerSize);
      for (int l = (layerSize - 1); l >= 0; l--) {
        errorOfHiddenNode[l].resize(hiddenLayerSize);

        for (size_t node = 0; node < hiddenLayerSize; node++) {

          double temp = 0;
          if (l == (layerSize - 1)) {
            // last layer
            for (size_t x = 0; x < outputLayerSize; x++) {
              temp += outputLayerWeights[node][x] * errorOfOutputLayer[x];
            }
            errorOfHiddenNode[l][node] =
                temp * gprime(activationHidden[l][node]);
          } else {
            for (size_t x = 0; x < hiddenLayerSize; x++) {
              temp += hiddenLayerWeights[l + 1][node][x] *
                      errorOfHiddenNode[l + 1][x];
            }
            errorOfHiddenNode[l][node] =
                temp * gprime(activationHidden[l][node]);
          }
        }
      }
      // Adjust weights for hidden layers
      for (int i = layerSize - 1; i >= 0; i--) {
        for (int j = 0; j < hiddenLayerWeights[i].size(); j++) {
          for (int z = 0; z < hiddenLayerSize; z++) {
            if (i == 0) {
              hiddenLayerWeights[i][j][z] +=
                  alpha * activationInput[j] * errorOfHiddenNode[i][z];
            } else {
              hiddenLayerWeights[i][j][z] +=
                  alpha * activationHidden[i - 1][j] * errorOfHiddenNode[i][z];
            }
          }
        }
      }
    }

    // Now run the validation set without adjusting the weights
    double validation_error = 0;
    double correctValidation = 0;
    for (size_t example = 0; example < validationexamples; example++) {

      // propagate the inputs forward to compute the outputs
      vector<double> activationInput(
          inputLayerSize); // We store the activation of each node (over all
                           // input and hidden layers) as we need that data
                           // during back propagation.
      // initialize input layer with training data
      for (size_t i = 0; i < inputLayerSize; i++) {
        activationInput[i] = validation_set[example][i];
      }

      vector<vector<double>> activationHidden(layerSize);
      // calculate activations of hidden layers
      for (size_t layer = 1; layer <= layerSize; layer++) {
        // Loops for every node at layer i
        activationHidden[layer - 1].resize(hiddenLayerSize);
        for (size_t j = 0; j < hiddenLayerSize; j++) {
          // We are the first later => look at the input activation
          double inputToHid = 0;
          if (layer == 1) {
            for (size_t i = 0; i < inputLayerSize; i++) {
              inputToHid +=
                  hiddenLayerWeights[layer - 1][i][j] * activationInput[i];
            }
          } else {
            for (size_t i = 0; i < hiddenLayerSize; i++) {
              inputToHid += hiddenLayerWeights[layer - 1][i][j] *
                            activationHidden[layer - 2][i];
            }
          }
          activationHidden[layer - 1][j] = g(inputToHid);
        }
      }

      // activation output
      vector<double> activationOutput(outputLayerSize);
      double temp = 0;
      for (size_t j = 0; j < outputLayerSize; j++) {
        for (size_t x = 0; x < hiddenLayerSize; x++) {
          temp += outputLayerWeights[x][j] *
                  activationHidden[activationHidden.size() - 1][x];
        }
        activationOutput[j] = g(temp);
        temp = 0;
      }

      // convert to binary
      vector<int> actualOutput(4);
      int number = validation_labels[example];
      for (int i = 0; number > 0; i++) {
        actualOutput[i] = number % 2;
        number = number / 2;
      }
      double tt = 0;
      for (size_t j = 0; j < outputLayerSize; j++) {
        tt += (activationOutput[j] - actualOutput[j]) *
              (activationOutput[j] - actualOutput[j]);
      }
      validation_error += tt;
      double activationOutputGuess = 0;
      // Convert the binary to a number
      for (int i = 0; i < 4; i++) {
        if (i == 0) {
          if (activationOutput[i] >= 0.9) {
            activationOutputGuess += 1;
          }
        } else if (i == 1) {
          if (activationOutput[i] >= 0.9) {
            activationOutputGuess += 2;
          }
        } else if (i == 2) {
          if (activationOutput[i] >= 0.9) {
            activationOutputGuess += 4;
          }
        } else {
          if (activationOutput[i] >= 0.9) {
            activationOutputGuess += 8;
          }
        }
      }
      if (validation_labels[example] == activationOutputGuess) {
        correctValidation++;
      }
    }

    // Print the data
    cout << "Training loss: "
         << ((double)training_error / trainingexamples) * 100 << "%" << endl;
    cout << "Training accuracy: "
         << ((double)correctTrainGuesses / trainingexamples) * 100 << "%"
         << endl;
    cout << "Validation loss: "
         << ((double)validation_error / validationexamples) * 100 << "%"
         << endl;
    cout << "Validation accuracy: "
         << ((double)correctValidation / validationexamples) * 100 << "%"
         << endl;
  }

  return;
}