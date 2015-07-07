#include "SupervisedSequence.h"
#include <Eigen/Dense>
#include <iostream>

using namespace std;

int main()
{
  vector<Eigen::VectorXd> inp;
  vector<Eigen::VectorXd> trgt;

  Eigen::VectorXd v(3);
  v(0) = 2;
  v(1) = 4;
  v(2) = 7;

  inp.push_back(v);
  trgt.push_back(2*v);

  v(0) = 4;
  v(1) = 19;
  v(2) = 74;

  inp.push_back(v);
  trgt.push_back(2*v);

  SupervisedSequence s(inp, trgt);


  for(int i=0; i<2; i++)
  {
    cout << "Input:" << endl << s.getInputAt(i) << endl;
    cout << "Target:" << endl << s.getTargetAt(i) << endl;
  }
}
