/**
* \class SoftplusLayer
*
* \brief A layer of Softplus units.
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

#ifndef __SOFTPLUSLAYER_H__
#define __SOFTPLUSLAYER_H__

#include "Layer.h"


class SoftplusLayer : public Layer
{
  public:
    void activate();
    Eigen::VectorXd gradient();
    SoftplusLayer(int size);
    ~SoftplusLayer();
};

#endif
