/**
* \class Teacher
*
* \brief A generic teacher 
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


#include <vector>

#ifndef __GRADIENTDESCENTTEACHER_H__
#define __GRADIENTDESCENTTEACHER_H__

#include "Teacher.h"
#include "SupervisedData.h"
#include "FeedForwardNeuralNetwork.h"
#include "StoppingCriteria.h"

using namespace std;

class StoppingCriteria;

class GradientDescentTeacher
{
  public:
    GradientDescentTeacher(FeedForwardNeuralNetwork* network, vector <SupervisedData*> dataset);
    ~GradientDescentTeacher();

    void train();

    void setStoppingCriteria(StoppingCriteria* criteria);

  private:
    FeedForwardNeuralNetwork* model;
    vector<SupervisedData*> dataset;

    vector<Connection*> connections;
    vector<Bias*> biases;

    vector<Eigen::MatrixXd> connectionGradients;
    vector<Eigen::MatrixXd> biasGradients;

    StoppingCriteria* stoppingCriteria;
};

#endif
