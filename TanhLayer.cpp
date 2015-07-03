/**
*
*/

#include "TanhLayer.h"

#include <Eigen/Dense>
#include <math.h>

TanhLayer::TanhLayer(int size) : Layer(size)
{

}

TanhLayer::~TanhLayer()
{

}

void TanhLayer::activate()
{
  for(int i=0; i<this->getSize(); i++)
  {
    this->activations(i) = (1.0 - exp(-this->net_input(i))) / (1.0 + exp(-this->net_input(i)));
  }
}


Eigen::VectorXd TanhLayer::gradient()
{
  Eigen::VectorXd grad(this->getSize());

  for(int i=0; i<this->getSize(); i++)
  {
    grad(i) = 1.0 - this->activations(i) * this->activations(i);
  }

  return grad;
}


