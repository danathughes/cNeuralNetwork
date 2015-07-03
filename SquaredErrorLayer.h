/**
* \class SquaredErrorLayer
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

#ifndef __SQUAREDERRORLAYER_H__
#define __SQUAREDERRORLAYER_H__

class SquaredErrorLayer : public ObjectiveLayer
{
  public:
    SquaredErrorLayer(int size, Layer* target);
    ~SquaredErrorLayer();

    void activate();
    Eigen::VectorXd gradient();

    void backprop();
  
    double cost();

  private:
    Layer* target;
};

#endif
