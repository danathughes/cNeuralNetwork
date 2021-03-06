/**
* \class Bias
*
* \brief A connection to provide bias to layers.
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

#include "Layer.h"
#include "Connection.h"

#ifndef __BIAS_H__
#define __BIAS_H__

class Bias : public Connection
{
  public:
    /**
     * Create a bias connection to the provided layer.  
     */
    Bias(Layer* outLayer);

    /**
     * Destroy the bias connection
     */
    ~Bias();

    void randomize();

    Eigen::VectorXd getBias();

    Eigen::MatrixXd getGradient();
    void updateWeights(Eigen::MatrixXd update);
    int getSize();


    Eigen::VectorXd getNetOutput();
    Eigen::VectorXd backpropDelta();
    int* getDimensions();

  private:
    int size;
    Eigen::VectorXd* weights;
    Layer* outLayer;
};

#endif
