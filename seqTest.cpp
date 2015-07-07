#include "Sequence.h"
#include "SupervisedData.h"
#include <Eigen/Dense>
#include <iostream>

using namespace std;

int main()
{
  Eigen::VectorXd inp(3);
  Eigen::VectorXd tar(2);

  vector<SupervisedData> seq;

  inp(0) = 2;
  inp(1) = 4;
  inp(2) = 7;

  tar(0) = 6;
  tar(1) = 11;

  cout << "Data 0:\t" << inp.transpose() << "\t\t-->\t" << tar.transpose() << endl;

  seq.push_back(SupervisedData(inp, tar));


  inp(0) = 4;
  inp(1) = 19;
  inp(2) = 74;

  tar(0) = 29;
  tar(1) = 104;

  cout << "Data 1:\t" << inp.transpose() << "\t-->\t" << tar.transpose() << endl;

  seq.push_back(SupervisedData(inp, tar));


  inp(0) = 4;
  inp(1) = -50;
  inp(2) = 2;

  tar(0) = -17;
  tar(1) = 56;

  cout << "Data 2:\t" << inp.transpose() << "\t-->\t" << tar.transpose() << endl;

  seq.push_back(SupervisedData(inp, tar));


  inp(0) = 20;
  inp(1) = -10;
  inp(2) = -5;

  tar(0) = -7;
  tar(1) = 41;

  cout << "Data 3:\t" << inp.transpose() << "\t-->\t" << tar.transpose() << endl;

  seq.push_back(SupervisedData(inp, tar));


  Sequence s(seq);

  cout << "Testing Sequence" << endl;

  cout << "  Sequence has " << s.getLength() << " items" << endl;

  for(int i=0; i<s.getLength(); i++)
  {
    cout << "    " << i << ":\t" << s.getDataAt(i).getInput().transpose() << "\t-->\t" << s.getDataAt(i).getTarget().transpose() << endl;
  }

  cout << endl;

  cout << "  Same thing with iterator:" << endl;

  while(s.hasNext())
  {
    SupervisedData data = s.next();
    cout << "    " << data.getInput().transpose() << "\t-->\t" << data.getTarget().transpose() << endl;
  }

  cout << endl;

  cout << "  Again with iterator:" << endl;

  s.reset();

  while(s.hasNext())
  {
    SupervisedData data = s.next();
    cout << "    " << data.getInput().transpose() << "\t-->\t" << data.getTarget().transpose() << endl;
  }

}

