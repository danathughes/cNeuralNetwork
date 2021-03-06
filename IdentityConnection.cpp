/**
* 
*/

#include "IdentityConnection.h"

#include <Eigen/Dense>

#include <stdlib.h>

IdentityConnection::IdentityConnection(Layer* inLayer, Layer* outLayer) : Connection()
{
  this->inLayer = inLayer;
  this->outLayer = outLayer;

  this->dim[0] = this->outLayer->getSize();
  this->dim[1] = this->inLayer->getSize();

  this->weights = new Eigen::MatrixXd(dim[0], dim[1]);

  for(int i=0; i<this->dim[0]; i++)
  {
    for(int j=0; j<this->dim[1]; j++)
    {
      (*(this->weights))(i,j) = 0.0;
    }
  }

  for(int i=0; i<this->dim[0]; i++)
  {
    (*(this->weights))(i,i) = 1.0;
  }

  outLayer->addInputConnection(this);
  inLayer->addOutputConnection(this);
}


IdentityConnection::~IdentityConnection()
{
  // Free up the weight matrix memory
  delete this->weights;
}


int* IdentityConnection::getDimensions()
{
  return dim;
}

void IdentityConnection::randomize()
{
  // Do nothing!  It's an identity matrix!!! LOL!!! 
}


Eigen::VectorXd IdentityConnection::getNetOutput()
{
  return this->inLayer->getOutput();
}


Eigen::VectorXd IdentityConnection::backpropDelta()
{
  return this->outLayer->getDeltas();
}


Eigen::MatrixXd IdentityConnection::getGradient()
{
  // Just need a dummy value to fulfill the interface
  return Eigen::MatrixXd(dim[0], dim[1]);
}


void IdentityConnection::updateWeights(Eigen::MatrixXd update)
{
  // Don't need to do anything...
}
