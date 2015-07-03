/**
* \class AndStoppingCriteria
*
* \brief A stopping criteria which stops when all member criteria are ready to stop
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
#include <vector>

#ifndef __ANDSTOPPINGCRITERIA_H__
#define __ANDSTOPPINGCRITERIA_H__

using namespace std;

class Teacher;

class AndStoppingCriteria : public StoppingCriteria
{
  public:
    AndStoppingCriteria(Teacher* trainer);
    ~AndStoppingCriteria();
    void addCriteria(StoppingCriteria* criteria);
    virtual void update();
    virtual void reset();
    virtual bool stop()=0;
  private:
    vector<StoppingCriteria*> criteria;
    Teacher* trainer;
    int iterationNumber;
};

#endif
