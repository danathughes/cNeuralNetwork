/**
* \class SoftmaxLayer
*
* \brief A layer of linear units.
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

#include "Layer.h"
#include <Eigen/Dense>

#ifndef __SOFTMAXLAYER_H__
#define __SOFTMAXLAYER_H__

class SoftmaxLayer : public Layer
{
  public:
    void activate();
    Eigen::VectorXd gradient();
    SoftmaxLayer(int size);
    ~SoftmaxLayer();
};

#endif
