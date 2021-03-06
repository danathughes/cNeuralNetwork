/**
* 
*/

#include "Bias.h"

#include <Eigen/Dense>

#include <stdlib.h>

using namespace std;

Bias::Bias(Layer* outLayer) : Connection()
{
  // Connect this to the corresponding layer
  this->outLayer = outLayer;
  this->size = this->outLayer->getSize();

  // Create the vector of biases
  this->weights = new Eigen::VectorXd(this->size);

  outLayer->addBias(this);
}


Bias::~Bias()
{
  // Free up the weight vector memory
  delete this->weights;
}


Eigen::VectorXd Bias::getBias()
{
  return *(this->weights);
}


void Bias::randomize()
{
  for(int i=0; i<this->size; i++)
    (*(this->weights))(i) = 0.001 * ((double) rand()) / RAND_MAX;
}


Eigen::MatrixXd Bias::getGradient()
{
  return -this->outLayer->getDeltas();
}


void Bias::updateWeights(Eigen::MatrixXd update)
{
  for(int i=0; i<this->size; i++)
    (*(this->weights))(i) += update(i);
}


int Bias::getSize()
{
  return size;
}


int* Bias::getDimensions()
{
  int dim[2];
  dim[0] = this->size;
  dim[1] = 1;
}


Eigen::VectorXd Bias::getNetOutput()
{
  return this->getBias();
}


Eigen::VectorXd Bias::backpropDelta()
{
  // Unneeded operation - just to maintain interface
  return Eigen::VectorXd(1);
}

