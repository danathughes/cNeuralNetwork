/**
* \class Sequence
*
* \brief A sequence of data
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


#include "Sequence.h"

Sequence::Sequence(vector<SupervisedData> sequence)
{
  this->sequence = sequence;
  this->currentIndex = 0;
}


Sequence::~Sequence()
{

}


int Sequence::getLength()
{
  return this->sequence.size();
}


// Just get the value function
SupervisedData Sequence::getDataAt(int idx)
{
  return this->sequence.at(idx);
}


// Iterator functions
void Sequence::reset()
{
  this->currentIndex = 0;
}


bool Sequence::hasNext()
{
  return this->currentIndex < this->sequence.size();
}


SupervisedData Sequence::next()
{
  SupervisedData data = this->getDataAt(currentIndex);
  currentIndex += 1;
  return data;  
}
