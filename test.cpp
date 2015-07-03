#include "SigmoidLayer.h"
#include "LinearLayer.h"
#include "SoftmaxLayer.h"
#include "FullConnection.h"
#include "BiasConnection.h"
#include "FeedForwardNeuralNetwork.h"
#include "SquaredErrorLayer.h"
#include "IdentityConnection.h"

#include <iostream>

using namespace std;

int main()
{
  cout << "Running a test!" << endl;

  cout << "Creating Layers" << endl;
  Layer* in = new LinearLayer(2);
  Layer* l1 = new SigmoidLayer(2);
  Layer* out = new SigmoidLayer(1);

  cout << "Creating connections and biases" << endl;
  FullConnection* f1 = new FullConnection(in, l1);
  BiasConnection* b1 = new BiasConnection(l1);

  FullConnection* f2 = new FullConnection(l1, out);
  BiasConnection* b2 = new BiasConnection(out);

  cout << "Creating neural network" << endl;

  FeedForwardNeuralNetwork nn;
  nn.addInputLayer(in);
  nn.addLayer(l1);
  nn.addOutputLayer(out);
  nn.addConnection(f1);
  nn.addBias(b1);
  nn.addConnection(f2);
  nn.addBias(b2);

  Layer* target = new LinearLayer(1);
  Layer* objective = new SquaredErrorLayer(1, target);

  IdentityConnection* id1 = new IdentityConnection(out, objective);
  
  nn.addLayer(objective);
  nn.addConnection(id1);
 
  cout << "Randomizing the weights" << endl;
  f1->randomize();
  f2->randomize();
  b1->randomize();
  b2->randomize();

  // Setting the weights explicitly
  (*(f1->weights))(0,0) = 0.1;
  (*(f1->weights))(0,1) = 0.2;
  (*(f1->weights))(1,0) = 0.3;
  (*(f1->weights))(1,1) = 0.4;

  (*(f2->weights))(0,0) = 0.1;
  (*(f2->weights))(0,1) = 0.2;

  (*(b1->weights))(0,0) = 0.1;
  (*(b1->weights))(1,0) = 0.2;

  (*(b2->weights))(0,0) = 0.3;

  cout << "Input->Hidden weights: " << endl << *(f1->weights) << endl;
  cout << "Hidden->output weights: " << endl << *(f2->weights) << endl;
  cout << "Hidden->bias: " << endl << *(b1->weights) << endl;
  cout << "Output->bias: " << endl << *(b2->weights) << endl;


//  cout << "Performing forward pass on (0,1)" << endl;

  Eigen::VectorXd input(2);


  for(int i=0; i<4; i++)
  {
    cout << i << "\t";

    input(0) = 0.0;
    input(1) = 0.0;

    nn.setInput(input);
    nn.forward();

    cout << out->getOutput() << "\t";
     

    input(0) = 0.0;
    input(1) = 1.0;

    nn.setInput(input);
    nn.forward();

    cout << out->getOutput() << "\t";


    input(0) = 1.0;
    input(1) = 0.0;

    nn.setInput(input);
    nn.forward();

    cout << out->getOutput() << "\t";


    input(0) = 1.0;
    input(1) = 1.0;

    nn.setInput(input);
    nn.forward();

    cout << out->getOutput() << endl;

   // cout << "Input: " << endl << in->getOutput() << endl;

//    cout << "Performing activations..." << endl;

//    nn.forward();

//    cout << "Done!" << endl;

//    cout << "Input: " << endl << in->getOutput() << endl;
 //   cout << "Hidden Layer: " << endl << l1->getOutput() << endl;
 //   cout << "Output Layer: " << endl << out->getOutput() << endl;

//    cout << "Cost = " << ((SquaredErrorLayer*) objective) -> cost() << endl;

  //  cout << "Backprop time!" << endl;

  //  nn.backward(); 

  // cout << "Input delta: " << endl << in->deltas << endl;
  //  cout << "Hidden delta: " << endl << l1->deltas << endl;
  //  cout << "Output delta: " << endl << out->deltas << endl;
  //  cout << "Cost delta: " << endl << objective->deltas << endl;

  //  cout << "WEIGHT GRADIENTS" << endl;
  //  cout << "Input->Hidden" << endl << f1->getGradient() << endl;
  //  cout << "Hidden->Output" << endl << f2->getGradient() << endl;
  //  cout << "Hidden Bias" << endl << b1->getGradient() << endl;
  //  cout << "Output Bias" << endl << b2->getGradient() << endl;
  }
/*
  cout << "Updating Weights: " << endl;
  cout << "  Input->Hidden Weights...";
  f1->updateWeights(-0.5*f1->getGradient());
  cout << "Done!" << endl;
  cout << "  Hidden Bias...";
  b1->updateWeights(-0.5*b1->getGradient());
  cout << "Done!" << endl;
  cout << "  Hidden->Output Weignts...";
  f2->updateWeights(-0.5*f2->getGradient());
  cout << "Done!" << endl;
  cout << "  Output Bias...";
  b2->updateWeights(-0.5*b2->getGradient());

  cout << "Round 2!" << endl;

  nn.forward();

  cout << "Done!" << endl;

  cout << "Input: " << endl << in->getOutput() << endl;
  cout << "Hidden Layer: " << endl << l1->getOutput() << endl;
  cout << "Output Layer: " << endl << out->getOutput() << endl;

  cout << "Cost = " << ((SquaredErrorLayer*) objective) -> cost() << endl;

*/
  cout << "Done!" << endl;

  return 0;
}
