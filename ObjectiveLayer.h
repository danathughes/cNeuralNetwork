#ifndef __OBJECTIVE_H__
#define __OBJECTIVE_H__

#include "Layer.h"

class ObjectiveLayer : public Layer
{
  public:
    ObjectiveLayer(int size);
    virtual double cost()=0;
};

#endif
