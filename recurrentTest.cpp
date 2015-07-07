#include "SigmoidLayer.h"
#include "LinearLayer.h"
#include "SoftmaxLayer.h"
#include "TanhLayer.h"
#include "FullConnection.h"
#include "Bias.h"
#include "FeedForwardNeuralNetwork.h"
#include "SquaredErrorLayer.h"
#include "IdentityConnection.h"
#include "SupervisedData.h"
#include "Teacher.h"
#include "MaxIterationStoppingCriteria.h"
#include "RecurrentLayer.h"

#include <iostream>

using namespace std;


int main()
{
  cout << "Running a test!" << endl;

  cout << "Creating a recurrent linear layer" << endl;

  LinearLayer* input = new LinearLayer(2);
  RecurrentLayer* rl = new RecurrentLayer(new LinearLayer(2));

  cout << "Setting the input to (-1,1), and the recurrent input to (0,0)" << endl;

  Eigen::VectorXd inp = Eigen::VectorXd(2);
  Eigen::VectorXd rec = Eigen::VectorXd(2);

  inp(0) = -1.0;
  inp(1) = 1.0;
  rec(0) = 0.0;
  rec(1) = 0.0;

  input->setInput(inp);
  input->activate();

  rl->setRecurrentInput(rec);

  IdentityConnection* conn = new IdentityConnection(input, rl);

  cout << "Input:" << endl << input->getInput() << endl;
  cout << "Recurrent Input:" << endl << rl->getRecurrentInput() << endl;

  cout << "Activating!  Output should be the sum of the inputs" << endl;
  rl->calculateNetInput();
  rl->activate();

  cout << "Output: " << endl << rl->getOutput() << endl;

  cout << "Making a recurrent connection..." << endl;
  IdentityConnection* recconn = new IdentityConnection(rl, rl->getRecurrentConnection());  

  cout << "Performing a time step." << endl;
  rl->step();

  inp(0) = -0.5;
  inp(1) = 1.5;

  input->setInput(inp);
  input->activate();


  cout << "Activating again.  Output should be input + recurrent -> (-2,2)" << endl;
  rl->calculateNetInput();
  rl->activate();

  cout << "Output: " << endl << rl->getOutput() << endl;

  cout << "Done!" << endl;

  return 0;
}
