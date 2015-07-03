/**
* NeuralNetwork.h
* Dana Hughes
* version 1.0
* 15-May-2015
* 
* Fully connected neural network model, which predicts an output vector given
* an input vector.
*
* Revisions:
*   1.0		Initial version, ported from the python version.
*/

#ifndef __NEURALNETWORK_H__
#define __NEURALNETWORK_H__

#include <vector>
#include <Eigen/Dense>

#include "CostFunction.h"
#include "ActivationFunction.h"

using namespace std;

/**
*
*/
class NeuralNetwork
{
  private:
    int numInputs;
    int numOutputs;
    int numLayers;
    vector<int> layerSizes;

    // Weight matrices and biases
    vector<Eigen::MatrixXd> weights;
    vector<Eigen::VectorXd> biases;

    vector<ActivationFunction> activationFunctions;
    CostFunction costFunction;



  public:
    NeuralNetwork(vector<int>, vector<ActivationFunction>, CostFunction); 
    void randomizeWeights();
    double cost(Eigen::MatrixXd, Eigen::MatrixXd);
    vector<Eigen::MatrixXd> gradient(Eigen::MatrixXd, Eigen::MatrixXd);
    vector<Eigen::MatrixXd> getWeights();
    void updateWeights(vector<Eigen::MatrixXd>);
    vector<Eigen::VectorXd> activate(Eigen::VectorXd);

    Eigen::VectorXd activation_function(Eigen::VectorXd);
    Eigen::VectorXd gradient_function(Eigen::VectorXd);
    double cost_function(Eigen::VectorXd, Eigen::VectorXd);
    Eigen::VectorXd cost_gradient(Eigen::VectorXd, Eigen::VectorXd);

    Eigen::VectorXd predict(Eigen::VectorXd);
    Eigen::VectorXi classify(Eigen::VectorXd);


};

#endif
