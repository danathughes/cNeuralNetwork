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

#ifndef __TEACHER_H__
#define __TEACHER_H__

#include "SupervisedData.h"
#include "FeedForwardNeuralNetwork.h"
#include "StoppingCriteria.h"

using namespace std;

class StoppingCriteria;

class Teacher
{
  public:
    Teacher(FeedForwardNeuralNetwork* network, vector <SupervisedData*> dataset);
    ~Teacher();

    void train();

    void setStoppingCriteria(StoppingCriteria* criteria);

  private:
    FeedForwardNeuralNetwork* model;
    vector<SupervisedData*> dataset;

    vector<Connection*> connections;
    vector<Eigen::MatrixXd> connectionGradients;

    StoppingCriteria* stoppingCriteria;
};

#endif
