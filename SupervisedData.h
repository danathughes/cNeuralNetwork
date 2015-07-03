/**
* \class SupervisedData
*
* \brief An instance of supervised data
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

#ifndef __SUPERVISEDDATA_H__
#define __SUPERVISEDDATA_H__

#include <Eigen/Dense>
#include <vector>

using namespace std;

class SupervisedData
{
  public:
    SupervisedData(Eigen::VectorXd input, Eigen::VectorXd target);
    ~SupervisedData();

    Eigen::VectorXd getInput();
    Eigen::VectorXd getTarget();
  private:
    Eigen::VectorXd input;
    Eigen::VectorXd target;
};

#endif
