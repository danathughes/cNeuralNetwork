/**
* \class IdentityConnection
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

#ifndef __IDENTITYCONNECTION_H__
#define __IDENTITYCONNECTION_H__

class IdentityConnection : public Connection
{
  public:
    /**
     * Create a full connection between the two provided layers.  
     */
    IdentityConnection(Layer* inLayer, Layer* outLayer);

    /**
     * Destroy the full connection
     */
    ~IdentityConnection();

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
