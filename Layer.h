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

#include <Eigen/Dense>
#include <string>
#include <vector>

#ifndef __LAYER_H__
#define __LAYER_H__

using namespace std;

class Bias;
class Connection;

class Layer
{
  public:
    Layer(int size);
    ~Layer();

    /**
     * Return the number of units in this layer.
     */
    int getSize();

    virtual void activate() = 0;
    virtual Eigen::VectorXd gradient() = 0;

    void setInput(Eigen::VectorXd input);
    void setOutput(Eigen::VectorXd output);
    void setDelta(Eigen::VectorXd delta);
    Eigen::VectorXd getInput();
    Eigen::VectorXd getOutput();
    Eigen::VectorXd getDeltas();
    void clearDeltas();

    void addInputConnection(Connection* connection);
    void addOutputConnection(Connection* connection);
    void addBias(Bias* bias);

    void setName(string name);
    string getName();

    vector<Connection*> getInputConnections();
    vector<Bias*> getBiases();

    void calculateNetInput();

    virtual void backprop();

  protected:
    int size;
    Eigen::VectorXd net_input;
    Eigen::VectorXd activations;
    Eigen::VectorXd deltas;

    vector<Connection*> inputConnections;
    vector<Connection*> outputConnections;
    vector<Bias*> biases;

    string name;
};

#endif
