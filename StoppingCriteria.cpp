/**
* \class StoppingCriteria
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

#include "StoppingCriteria.h"
#include "Teacher.h"

StoppingCriteria::StoppingCriteria(Teacher* teacher)
{
  this->trainer = teacher;
  this->iterationNumber = 0;
}


StoppingCriteria::~StoppingCriteria()
{

}

void StoppingCriteria::update()
{
  this->iterationNumber += 1;
}

void StoppingCriteria::reset()
{
  this->iterationNumber = 0;
}

