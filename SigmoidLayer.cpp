/**
*
*/

#include "SigmoidLayer.h"

#include <Eigen/Dense>
#include <math.h>

//#include <iostream>

using namespace std;

SigmoidLayer::SigmoidLayer(int size) : Layer(size)
{

}

SigmoidLayer::~SigmoidLayer()
{

}

void SigmoidLayer::activate()
{
//  cout << "About to activate..." << endl << this->net_input << endl;

  for(int i=0; i<this->getSize(); i++)
  {
    this->activations(i) = 1.0 / (1.0 + exp(-this->net_input(i)));
  }
}


Eigen::VectorXd SigmoidLayer::gradient()
{
  Eigen::VectorXd grad(this->getSize());

  for(int i=0; i<this->getSize(); i++)
  {
    grad(i) = this->activations(i) * (1.0 - this->activations(i));
  }

  return grad;
}


