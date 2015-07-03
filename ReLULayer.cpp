/**
*
*/

#include "ReLULayer.h"

#include <Eigen/Dense>
#include <math.h>

ReLULayer::ReLULayer(int size) : Layer(size)
{

}

ReLULayer::~ReLULayer()
{

}

void ReLULayer::activate()
{
  for(int i=0; i<this->getSize(); i++)
  {
    if(this->net_input(i) > 0.0)
    {
      this->activations(i) = net_input(i);
    }
    else
    {
      this->activations(i) = 0.0;
    }
  }
}


Eigen::VectorXd ReLULayer::gradient()
{
  Eigen::VectorXd grad(this->getSize());

  for(int i=0; i<this->getSize(); i++)
  {
    if(this->net_input(i) > 0.0)
    {
      grad(i) = 1.0;
    }
    else
    {
      grad(i) = 0.0;
    }
  }

  return grad;
}


