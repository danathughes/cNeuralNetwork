/**
* \class MaxIterationStoppingCriteria
*
* \brief Ends training when a maximum number of iterations is reached.
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

#ifndef __MAXITERATIONSTOPPINGCRITERIA_H__
#define __MAXITERATIONSTOPPINGCRITERIA_H__

class Teacher;

class MaxIterationStoppingCriteria : public StoppingCriteria
{
  public:
    MaxIterationStoppingCriteria(Teacher* trainer, int maxIterations);
    ~MaxIterationStoppingCriteria();
    void reset();
    void update();
    bool stop();
  private:
    Teacher* trainer;
    int iterationNumber;
    int maxIterations;
};

#endif
