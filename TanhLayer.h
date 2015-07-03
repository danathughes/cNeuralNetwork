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

#ifndef __TANHLAYER_H__
#define __TANHLAYER_H__

#include "Layer.h"


class TanhLayer : public Layer
{
  public:
    void activate();
    Eigen::VectorXd gradient();
    TanhLayer(int size);
    ~TanhLayer();
};

#endif
