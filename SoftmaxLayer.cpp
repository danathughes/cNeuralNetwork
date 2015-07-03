/**
*
*/

#include "SoftmaxLayer.h"

#include <Eigen/Dense>
#include <math.h>

using namespace std;

SoftmaxLayer::SoftmaxLayer(int size) : Layer(size)
{

}

SoftmaxLayer::~SoftmaxLayer()
{

}

void SoftmaxLayer::activate()
{
  double sum = 0.0;

  for(int i=0; i<this->getSize(); i++)
  {
    this->activations(i) = 1.0 / (1.0 + exp(-this->net_input(i)));
    sum += this->activations(i);
  }

  for(int i=0; i<this->getSize(); i++)
  {
    this->activations(i) /= sum;
  }
}


Eigen::VectorXd SoftmaxLayer::gradient()
{
  Eigen::VectorXd grad(this->getSize());

  for(int i=0; i<this->getSize(); i++)
  {
    grad(i) = 1.0;
  }

  return grad;
}


