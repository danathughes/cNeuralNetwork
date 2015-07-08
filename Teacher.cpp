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
#include "Teacher.h"

#include <iostream>

using namespace std;

Teacher::Teacher(FeedForwardNeuralNetwork* model, vector<SupervisedData*> dataset)
{
  this->dataset = dataset;
  this->model = model;

  this->connections = model->getConnections();

  vector<Eigen::MatrixXd> connectionGradients;

  for(int i=0; i<connections.size(); i++)
  {
    this->connectionGradients.push_back(connections.at(i)->getGradient());
  }
}


Teacher::~Teacher()
{

}


void Teacher::train()
{
  double cost = 0.0;
  double learningRate = 0.25;
  int numIterations = 0;

  this->stoppingCriteria->reset();

  while(!this->stoppingCriteria->stop())
  {
    cost = 0.0;

    for(int j=0; j<this->connectionGradients.size(); j++)
    {
      this->connectionGradients.at(j) *= 0;
    }

    cout << numIterations << ":\t";

    for(int j=0; j<this->dataset.size(); j++)
    {    
      this->model->setInput(this->dataset.at(j)->getInput());
      this->model->setTarget(this->dataset.at(j)->getTarget());

      this->model->forward();
      cost += this->model->getObjectiveLayer()->cost();
      this->model->backward(); 

      for(int k=0; k<this->connections.size(); k++)
      {
        this->connectionGradients.at(k) += this->connections.at(k)->getGradient();
      }

      cout << (this->model->getOutputLayer()->getOutput()).transpose() << "\t";
    }

    cout << cost / dataset.size() << endl;

    for(int j=0; j<this->connections.size(); j++)
    {
      this->connections.at(j)->updateWeights(-learningRate*this->connectionGradients.at(j));
    }

    numIterations++;
    this->stoppingCriteria->update();
  }
}


void Teacher::setStoppingCriteria(StoppingCriteria* criteria)
{
  this->stoppingCriteria = criteria;
}
