/**
* \class SupervisedSequence
*
* \brief An supervised sequence of data
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

#ifndef __SUPERVISEDSEQUENCE_H__
#define __SUPERVISEDSEQUENCE_H__

#include <Eigen/Dense>
#include <vector>

using namespace std;

class SupervisedSequence
{
  public:
    SupervisedSequence(vector<Eigen::VectorXd> input, vector<Eigen::VectorXd> target);
    ~SupervisedSequence();

    Eigen::VectorXd getInputAt(int idx);
    Eigen::VectorXd getTargetAt(int idx);
  private:
    vector<Eigen::VectorXd> input;
    vector<Eigen::VectorXd> target;
};

#endif
