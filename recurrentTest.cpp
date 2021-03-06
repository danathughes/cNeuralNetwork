#include "SigmoidLayer.h"
#include "LinearLayer.h"
#include "SoftmaxLayer.h"
#include "FullConnection.h"
#include "Bias.h"
#include "RecurrentNeuralNetwork.h"
#include "IdentityConnection.h"
#include "Teacher.h"
#include "MaxIterationStoppingCriteria.h"
#include "RecurrentLayer.h"
#include "Sequence.h"
#include "ObjectiveLayer.h"
#include "CrossEntropyErrorLayer.h"
#include "SquaredErrorLayer.h"
#include "SupervisedData.h"

#include <iostream>
#include <vector>
using namespace std;


Sequence createSequence()
{
  vector<SupervisedData> s;
  Eigen::VectorXd inp(3);
  Eigen::VectorXd tgt(2);

  inp(0) = 0.0;
  inp(1) = 0.0;
  inp(2) = 0.0;

  tgt(0) = 1.0;
  tgt(1) = 0.0;

  s.push_back(SupervisedData(inp, tgt));

  inp(0) = 0.0;
  inp(1) = 1.0;
  inp(2) = 0.0;

  tgt(0) = 1.0;
  tgt(1) = 0.0;

  s.push_back(SupervisedData(inp, tgt));

  inp(0) = 0.0;
  inp(1) = 0.0;
  inp(2) = 1.0;

  tgt(0) = 0.0;
  tgt(1) = 1.0;

  s.push_back(SupervisedData(inp, tgt));

  inp(0) = 1.0;
  inp(1) = 0.0;
  inp(2) = 1.0;

  tgt(0) = 0.0;
  tgt(1) = 1.0;

  s.push_back(SupervisedData(inp, tgt));

  Sequence seq(s);

  return seq;
}


RecurrentNeuralNetwork createRNN()
{
  RecurrentNeuralNetwork rnn;

  Layer* input = new LinearLayer(3);
  input->setName("input");
  RecurrentLayer* hidden = new RecurrentLayer(new SigmoidLayer(2));
  hidden->setName("hidden");
  Layer* output = new SoftmaxLayer(2);
  output->setName("output");
  Layer* target = new LinearLayer(2);
  target->setName("target");

  FullConnection* input_to_hidden = new FullConnection(input, hidden);
  FullConnection* hidden_to_output = new FullConnection(hidden, output);
  FullConnection* hidden_to_hidden = new FullConnection(hidden, hidden->getRecurrentConnection());
  Bias* hidden_bias = new Bias(hidden);
  Bias* output_bias = new Bias(output);

  input_to_hidden->randomize();
  hidden_to_output->randomize();
  hidden_to_hidden->randomize();
  hidden_bias->randomize();
  output_bias->randomize();

//  cout << "Wih:" << endl << input_to_hidden->getWeights() << endl;

  ObjectiveLayer* objective = new CrossEntropyErrorLayer(2, target);
  IdentityConnection* output_to_objective = new IdentityConnection(output, objective);

  rnn.addInputLayer(input);
  rnn.addLayer(hidden);
  rnn.addOutputLayer(output);
//  rnn.addLayer(target);
  rnn.addLayer(objective);

  rnn.setTargetLayer(target);
  rnn.setObjectiveLayer(objective);


  rnn.addConnection(input_to_hidden);
  rnn.addConnection(hidden_to_hidden);
  rnn.addConnection(hidden_to_output);
  rnn.addConnection(output_to_objective);
  rnn.addConnection(hidden_bias);
  rnn.addConnection(output_bias);

  return rnn;
}

int main()
{
  // Create a sequence
  cout << "Creating a sequence" << endl;
  Sequence seq = createSequence();

  cout << "Creating an RNN" << endl;
  RecurrentNeuralNetwork rnn = createRNN();

  cout << "Initial cost: " << rnn.cost(&seq) << endl;

  double learning_rate = 0.01;

  for(int i=0; i<100000; i++)
  {
    cout << i  << ": " << rnn.cost(&seq) << endl;
    vector<Eigen::VectorXd> output = rnn.output(&seq);
    for(int j=0; j<output.size(); j++)
    {
      cout << "  " << output.at(j).transpose() << endl;
    }

    vector<Eigen::MatrixXd> grad = rnn.getParameterGradients(&seq);

    for(int j=0; j<rnn.getConnections().size(); j++)
    {
      rnn.getConnections().at(j)->updateWeights(-learning_rate*grad.at(j));
    }
  }

 
  return 0;
}
