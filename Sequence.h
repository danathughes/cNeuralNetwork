/**
* \class Sequence
*
* \brief A sequence of supervised or unsupervised data
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

#ifndef __SEQUENCE_H__
#define __SEQUENCE_H__

#include "SupervisedData.h"

#include <Eigen/Dense>
#include <vector>

using namespace std;

class Sequence
{
  public:
    Sequence(vector<SupervisedData> sequence);
    ~Sequence();

    SupervisedData getDataAt(int idx);
    int getLength();

    void reset();
    bool hasNext();
    SupervisedData next();

  private:
    int currentIndex;
    vector<SupervisedData> sequence;
};

#endif
