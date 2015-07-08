/**
* \class FullConnection
*
* \brief A complete connection between two layers in a network.
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
#include "Connection.h"

#include <Eigen/Dense>

#ifndef __FULLCONNECTION_H__
#define __FULLCONNECTION_H__

class FullConnection : public Connection
{
  public:
    /**
     * Create a full connection between the two provided layers.  
     */
    FullConnection(Layer* inLayer, Layer* outLayer);

    /**
     * Destroy the full connection
     */
    ~FullConnection();

    void randomize();

    Eigen::VectorXd getNetOutput();
    Eigen::VectorXd backpropDelta();

    Eigen::MatrixXd getGradient();
    void updateWeights(Eigen::MatrixXd update);

    int* getDimensions();

  private:
    Layer* inLayer;
    Layer* outLayer;

    int dim[2];

    Eigen::MatrixXd* weights;
};

#endif
