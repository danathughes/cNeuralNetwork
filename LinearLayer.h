/**
* \class LinearLayer
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

#ifndef __LINEARLAYER_H__
#define __LINEARLAYER_H__

class LinearLayer : public Layer
{
  public:
    void activate();
    Eigen::VectorXd gradient();
    LinearLayer(int size);
    ~LinearLayer();
};

#endif
