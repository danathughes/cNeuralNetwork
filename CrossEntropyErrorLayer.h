/**
* \class CrossEntropyErrorLayer
*
* \brief Not sure if this is what I want to do, but let's call this the
*        neural network super duper class for now.
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

#include "ObjectiveLayer.h"

#include <Eigen/Dense>

#ifndef __CROSSENTROPYERRORLAYER_H__
#define __CROSSENTORPYERRORLAYER_H__

class CrossEntropyErrorLayer : public ObjectiveLayer
{
  public:
    CrossEntropyErrorLayer(int size, Layer* target);
    ~CrossEntropyErrorLayer();

    void activate();
    Eigen::VectorXd gradient();

    void backprop();
  
    double cost();

  private:
    Layer* target;
};

#endif
