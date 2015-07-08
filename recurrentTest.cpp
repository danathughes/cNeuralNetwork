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
#include "SupervisedData.h"

#include <iostream>
#include <vector>
using namespace std;


int main()
{
  // Create a sequence
  cout << "Creating a sequence" << endl;

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

  cout << "Creating an RNN" << endl;

  RecurrentNeuralNetwork rnn;

  cout << "Creating the layers and connections" << endl;

  Layer* input = new LinearLayer(3);
  RecurrentLayer* hidden = new RecurrentLayer(new LinearLayer(2));
  Layer* output = new SoftmaxLayer(2);

  Layer* target = new LinearLayer(2);

  FullConnection* input_to_hidden = new FullConnection(input, hidden);
  FullConnection* hidden_to_output = new FullConnection(hidden, output);
  FullConnection* hidden_to_hidden = new FullConnection(hidden, hidden->getRecurrentConnection());
  Bias* hidden_bias = new Bias(hidden);
  Bias* output_bias = new Bias(output);
/*
  input_to_hidden->randomize();
  hidden_to_output->randomize();
  hidden_to_hidden->randomize();
  hidden_bias->randomize();
  output_bias->randomize();
*/
  Eigen::MatrixXd Wih(2,3);
  Wih << 0.1, 0.2, 0.3,
         0.4, 0.5, 0.6;
  input_to_hidden->updateWeights(Wih);


  Eigen::MatrixXd Whh(2,2);
  Whh << 0.1, 0.2,
         0.2, 0.1;
  hidden_to_hidden->updateWeights(Whh);

  Eigen::MatrixXd Who(2,2);
  Who << 0.25, 0.75,
         0.75, 0.25;
  hidden_to_output->updateWeights(Who);


  ObjectiveLayer* objective = new CrossEntropyErrorLayer(2, target);
  IdentityConnection* output_to_objective = new IdentityConnection(output, objective);

  cout << "Adding the layers to the RNN" << endl;
  rnn.addInputLayer(input);
  rnn.addLayer(hidden);
  rnn.addOutputLayer(output);
//  rnn.addLayer(target);
  rnn.addLayer(objective);

  rnn.setTargetLayer(target);
  rnn.setObjectiveLayer(objective);

  cout << "Adding the connections to the RNN" << endl;

  rnn.addConnection(input_to_hidden);
  rnn.addConnection(hidden_to_output);
  rnn.addConnection(output_to_objective);

  cout << "Running the sequence through the rnn" << endl;
  rnn.getParameterGradients(&seq);



  return 0;
}
