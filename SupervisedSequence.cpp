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


#include "SupervisedSequence.h"

using namespace std;

SupervisedSequence::SupervisedSequence(vector<Eigen::VectorXd> input, vector<Eigen::VectorXd> target)
{
  for(int i=0; i<input.size(); i++)
  {
    this->input.push_back(input.at(i));
    this->target.push_back(target.at(i));
  }
}


SupervisedSequence::~SupervisedSequence()
{

}


Eigen::VectorXd SupervisedSequence::getInputAt(int idx)
{
  return this->input.at(idx);
}


Eigen::VectorXd SupervisedSequence::getTargetAt(int idx)
{
  return this->target.at(idx);
}


