/**
*
*/

#include "LinearLayer.h"

#include <Eigen/Dense>
#include <math.h>

using namespace std;

LinearLayer::LinearLayer(int size) : Layer(size)
{

}

LinearLayer::~LinearLayer()
{

}

void LinearLayer::activate()
{
  for(int i=0; i<this->getSize(); i++)
  {
    this->activations(i) = this->net_input(i);
  }
}


Eigen::VectorXd LinearLayer::gradient()
{
  Eigen::VectorXd grad(this->getSize());

  for(int i=0; i<this->getSize(); i++)
  {
    grad(i) = 1.0;
  }

  return grad;
}


