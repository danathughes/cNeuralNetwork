/**
* \class MaxIterationStoppingCriteria
*
* \brief An object responsible for determining when to end training
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

#include "MaxIterationStoppingCriteria.h"
#include "Teacher.h"

MaxIterationStoppingCriteria::MaxIterationStoppingCriteria(Teacher* teacher, int maxIterations) : StoppingCriteria(teacher)
{
  this->maxIterations = maxIterations;
}


MaxIterationStoppingCriteria::~MaxIterationStoppingCriteria()
{

}

void MaxIterationStoppingCriteria::update()
{
  iterationNumber += 1;
}

void MaxIterationStoppingCriteria::reset()
{
  iterationNumber = 0;
}


bool MaxIterationStoppingCriteria::stop()
{
  return this->iterationNumber >= this->maxIterations;
}

