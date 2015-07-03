/**
* \class SupervisedData
*
* \brief An instance of supervised data
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


#include "SupervisedData.h"

using namespace std;

SupervisedData::SupervisedData(Eigen::VectorXd input, Eigen::VectorXd target)
{
  this->input = Eigen::VectorXd(input.rows());
  for(int i=0; i<input.rows(); i++)
    this->input(i) = input(i);
  this->target = Eigen::VectorXd(target.rows());
  for(int i=0; i<target.rows(); i++)
    this->target(i) = target(i);
}


SupervisedData::~SupervisedData()
{

}


Eigen::VectorXd SupervisedData::getInput()
{
  return this->input;
}


Eigen::VectorXd SupervisedData::getTarget()
{
  return this->target;
}


