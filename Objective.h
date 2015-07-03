#ifndef __OBJECTIVE_H__
#define __OBJECTIVE_H__

class ObjectiveLayer : public Layer
{
  public:
    virtual void cost()=0;
}
