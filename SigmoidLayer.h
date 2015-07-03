/**
* \class SigmoidLayer
*
* \brief A layer of Sigmoid units.
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

#include <Eigen/Dense>

#ifndef __SIGMOIDLAYER_H__
#define __SIGMOIDLAYER_H__

#include "Layer.h"


class SigmoidLayer : public Layer
{
  public:
    void activate();
    Eigen::VectorXd gradient();
    SigmoidLayer(int size);
    ~SigmoidLayer();
};

#endif
