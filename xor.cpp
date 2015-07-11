#include "SigmoidLayer.h"
#include "LinearLayer.h"
#include "SoftmaxLayer.h"
#include "TanhLayer.h"
#include "FullConnection.h"
#include "Bias.h"
#include "FeedForwardNeuralNetwork.h"
#include "SquaredErrorLayer.h"
#include "CrossEntropyErrorLayer.h"
#include "IdentityConnection.h"
#include "SupervisedData.h"
#include "Teacher.h"
#include "MaxIterationStoppingCriteria.h"

#include <iostream>

using namespace std;

vector<SupervisedData*> createDataset()
{
  Eigen::VectorXd input(2);
  Eigen::VectorXd label(2);

  input(0) = 0.0;
  input(1) = 0.0;
  label(0) = 0.0;
  label(1) = 1.0;

  SupervisedData* d0 = new SupervisedData(input, label);

  input(0) = 0.0;
  input(1) = 1.0;
  label(0) = 1.0;
  label(1) = 0.0;

  SupervisedData* d1 = new SupervisedData(input, label);

  input(0) = 1.0;
  input(1) = 0.0;
  label(0) = 1.0;
  label(1) = 0.0;

  SupervisedData* d2 = new SupervisedData(input, label);

  input(0) = 1.0;
  input(1) = 1.0;
  label(0) = 0.0;
  label(1) = 1.0;

  SupervisedData* d3 = new SupervisedData(input, label);

  vector<SupervisedData*> dataset;
  dataset.push_back(d0);
  dataset.push_back(d1);
  dataset.push_back(d2);
  dataset.push_back(d3);

  return dataset;
}


FeedForwardNeuralNetwork createNetwork()
{
  cout << "Creating Layers" << endl;
  Layer* in = new LinearLayer(2);
  Layer* l1 = new TanhLayer(2);
  Layer* out = new SigmoidLayer(1);
//  Layer* out = new SigmoidLayer(2);

  cout << "Creating connections and biases" << endl;
  FullConnection* f1 = new FullConnection(in, l1);
  Bias* b1 = new Bias(l1);

  FullConnection* f2 = new FullConnection(l1, out);
  Bias* b2 = new Bias(out);

  Layer* target = new LinearLayer(1);
  ObjectiveLayer* objective = new SquaredErrorLayer(1, target);
//  Layer* target = new LinearLayer(2);
//  ObjectiveLayer* objective = new CrossEntropyErrorLayer(2, target);
  IdentityConnection* id1 = new IdentityConnection(out, objective);

  cout << "Creating neural network" << endl;

  FeedForwardNeuralNetwork nn;
  nn.addInputLayer(in);
  nn.addLayer(l1);
  nn.addOutputLayer(out);
  nn.addConnection(f1);
  nn.addBias(b1);
  nn.addConnection(f2);
  nn.addBias(b2);  
  nn.addLayer(objective);
  nn.addConnection(id1);

  nn.setTargetLayer(target);
  nn.setObjectiveLayer(objective);
 
  cout << "Randomizing the weights" << endl;

  cout << "f1" << endl;
  f1->randomize();
  cout << "f2" << endl;
  f2->randomize();
  cout << "b1" << endl;
  b1->randomize();
  cout << "b2" << endl;
  b2->randomize();
  cout << "Done!" << endl;

  return nn;
}


int main()
{
  cout << "Running a test!" << endl;

  vector<SupervisedData*> dataset = createDataset();

  double cost = 0.0;

  FeedForwardNeuralNetwork nn = createNetwork();

  cout << "Neural network is made.  Time to make a teacher" << endl;

  Teacher trainer = Teacher(&nn, dataset);
  cout << "Done.  Setting Stopping criteria" << endl;
  trainer.setStoppingCriteria(new MaxIterationStoppingCriteria(&trainer, 10));

  cout << "Done!  Training!" << endl;
  trainer.train();

  cout << "Done!" << endl;

  return 0;
}
