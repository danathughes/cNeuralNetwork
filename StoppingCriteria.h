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

#ifndef __STOPPINGCRITERIA_H__
#define __STOPPINGCRITERIA_H__

class Teacher;

class StoppingCriteria
{
  public:
    StoppingCriteria(Teacher* trainer);
    ~StoppingCriteria();
    virtual void update();
    virtual void reset();
    virtual bool stop()=0;
  private:
    Teacher* trainer;
    int iterationNumber;
};

#endif
