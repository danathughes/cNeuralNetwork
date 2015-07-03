/**
* \class Connection
*
* \brief An interface for connections between two layers in a network.
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

#ifndef __CONNECTION_H__
#define __CONNECTION_H__

#include "Layer.h"

class Connection
{
  public:
    Connection(Layer* inLayer, Layer* outLayer) {};
    ~Connection() {};
    virtual Eigen::VectorXd getNetOutput()=0;
    virtual Eigen::VectorXd backpropDelta()=0;
    virtual Eigen::MatrixXd getGradient()=0;
    virtual void updateWeights(Eigen::MatrixXd update)=0;
    virtual int* getDimensions()=0;
  private:
};

#endif
