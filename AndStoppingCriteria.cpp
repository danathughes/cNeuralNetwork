/**
* \class AndStoppingCriteria
*
* \brief 
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

#include "AndStoppingCriteria.h"
#include "Teacher.h"

AndStoppingCriteria::AndStoppingCriteria(Teacher* teacher) : StoppingCriteria(teacher)
{

}


AndStoppingCriteria::~AndStoppingCriteria()
{

}


void AndStoppingCriteria::update()
{
  this->iterationNumber += 1;
  for(int i=0; i<this->criteria.size(); i++)
    this->criteria.at(i)->update();
}


void AndStoppingCriteria::reset()
{
  this->iterationNumber = 0;
  for(int i=0; i<this->criteria.size(); i++)
    this->criteria.at(i)->reset();
}


bool AndStoppingCriteria::stop()
{
  bool stop = true;
  for(int i=0; i<this->criteria.size(); i++)
    stop = stop && this->criteria.at(i)->stop();
  return stop;
}


void AndStoppingCriteria::addCriteria(StoppingCriteria* criteria)
{
  this->criteria.push_back(criteria);
}
