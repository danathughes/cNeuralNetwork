/**
* \class SquaredErrorLayer
*
* \brief Not sure if this is what I want to do, but let's call this the
*        neural network super duper class for now.
*
* \author $Author: dh$
*
* \version $Version: 1.0$
*
* \date $Date: 27-June-2015$
*
* Contact:  danathughes@gmail.com
*
* 
*/

#include <Eigen/Dense>
#include <iostream>

#include "Layer.h"
#include "SquaredErrorLayer.h"

using namespace std;

SquaredErrorLayer::SquaredErrorLayer(int size, Layer* target) : ObjectiveLayer(size)
{
  this->target = target;
}


SquaredErrorLayer::~SquaredErrorLayer()
{

}


void SquaredErrorLayer::activate()
{
  for(int i=0; i<this->getSize(); i++)
  {
    double error = this->target->getOutput()(i) - this->net_input(i);
    this->activations(i) = 0.5*error*error;
  }
}


Eigen::VectorXd SquaredErrorLayer::gradient()
{
  return this->target->getOutput() - this->net_input;
}


double SquaredErrorLayer::cost()
{
  return this->activations.sum();
}


void SquaredErrorLayer::backprop()
{
  Eigen::VectorXd grad = this->gradient();
  for(int i=0; i<this->getSize(); i++)
  {
    this->deltas(i) = grad(i);
  }
  cout << "---" << this->target->getOutput().transpose();
  cout << "---" << this->net_input.transpose();
  cout << "---" << this->getName() << ": " << grad.transpose() << endl;
}
