/**
* \class FeedForwardNeuralNetwork
*
* \brief An acylic, feed forward neural network.
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
#include "ObjectiveLayer.h"
#include "Connection.h"
#include "Bias.h"

#include <vector>


#ifndef __FEEDFORWARDNEURALNETWORK_H__
#define __FEEDFORWARDNEURALNETWORK_H__

using namespace std;

class FeedForwardNeuralNetwork
{
  public:
    FeedForwardNeuralNetwork();
    ~FeedForwardNeuralNetwork();

    void addInputLayer(Layer* layer);
    void addLayer(Layer* layer);
    void addOutputLayer(Layer* layer);
    void setTargetLayer(Layer* layer);
    void setObjectiveLayer(ObjectiveLayer* layer);
    void addConnection(Connection* connection);
    void addBias(Bias* bias);

    void forward();
    void backward();

    void setInput(Eigen::VectorXd input);
    void setTarget(Eigen::VectorXd target);

    vector<Layer*> getLayers();
    vector<Connection*> getConnections();
    vector<Bias*> getBiases();

    Layer* getInputLayer();
    Layer* getOutputLayer();
    ObjectiveLayer* getObjectiveLayer();

  private:
    vector<Layer*> layers;
    vector<Connection*> connections;
    vector<Bias*> biases;

    Layer* inputLayer;
    Layer* outputLayer;
    Layer* targetLayer;
    ObjectiveLayer* objectiveLayer;
};

#endif
