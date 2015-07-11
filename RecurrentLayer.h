/**
* \class Layer
*
* \brief A connection to provide bias to a layer in a network.
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

#include <vector>

#ifndef __RECURRENTLAYER_H__
#define __RECURRENTLAYER_H__

using namespace std;

class Bias;
class Connection;

class RecurrentLayer : public Layer
{
  public:
    RecurrentLayer(Layer* layer);
    ~RecurrentLayer();

    /**
     * Return the number of units in this layer.
     */
    void activate();
    Eigen::VectorXd gradient();

    void setRecurrentInput(Eigen::VectorXd input);
    Eigen::VectorXd getRecurrentInput();

    Layer* getRecurrentConnection();

    void step();
    void backstep();

//    Eigen::VectorXd getRecurrentInput();
//    void addRecurrentInputConnection(Connection* connection);
//    vector<Connection*> getRecurrentConnections();


  private:
    Layer* mainLayer;
    Layer* recurrentLayer;
};

#endif
