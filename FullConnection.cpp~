/**
* 
*/

#include "FullConnection.h"

#include <Eigen/Dense>

#include <stdlib.h>

#include <iostream>

using namespace std;

FullConnection::FullConnection(Layer* inLayer, Layer* outLayer) : Connection(inLayer, outLayer)
{
  this->inLayer = inLayer;
  this->outLayer = outLayer;

  this->dim[0] = this->outLayer->getSize();
  this->dim[1] = this->inLayer->getSize();

  this->weights = new Eigen::MatrixXd(dim[0], dim[1]);

  outLayer->addInputConnection(this);
  inLayer->addOutputConnection(this);
}


FullConnection::~FullConnection()
{
  // Free up the weight matrix memory
  delete this->weights;
}


void FullConnection::randomize()
{
  for(int i=0; i<this->dim[0]; i++)
    for(int j=0; j<this->dim[1]; j++)
      (*(this->weights))(i,j) = 0.5 * ((double) rand()) / RAND_MAX;
}


Eigen::VectorXd FullConnection::getNetOutput()
{
  return *(this->weights) * this->inLayer->getOutput();
}


Eigen::VectorXd FullConnection::backpropDelta()
{
  return (*(this->weights)).transpose() * this->outLayer->getDeltas();
}


Eigen::MatrixXd FullConnection::getGradient()
{
  Eigen::MatrixXd grad = - (this->outLayer->getDeltas()) * (this->inLayer->getOutput()).transpose();
  return grad;
}


void FullConnection::updateWeights(Eigen::MatrixXd update)
{
  for(int i=0; i<this->dim[0]; i++)
    for(int j=0; j<this->dim[1]; j++)
      (*(this->weights))(i,j) += update(i,j);
}

int* FullConnection::getDimensions()
{
  return dim;
}
