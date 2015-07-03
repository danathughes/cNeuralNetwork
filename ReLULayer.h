/**
* \class TanhLayer
*
* \brief A layer of Tanh units.
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

#ifndef __RELULAYER_H__
#define __RELULAYER_H__

#include "Layer.h"


class ReLULayer : public Layer
{
  public:
    void activate();
    Eigen::VectorXd gradient();
    ReLULayer(int size);
    ~ReLULayer();
};

#endif
