#ifndef VERIFICATOR_HPP
#define VERIFICATOR_HPP
#include "Utils.hpp"

//class for Verification task. Get new feature, then classify it based on DataBase

class Verificator
{
public:
  Verificator(struct Configuration& config);
  //load Model from Sklearn  for Face Verification
  void loadModel();
  //load verification data
  void verify();
  //compare if two features descrive same person or not
  int compare(cv::Mat& featureOne, cv::Mat& featureTwo);
  float predValue(cv::Mat featureOne, cv::Mat featureTwo);
  ~Verificator();


  //transform data from classification task to verfication
  void prepareVerificationData(cv::Mat& scaledFetures, cv::Mat& featuresVer
                                  ,std::vector<int>& labelsVecVer);
  //scale data
  void scaleData(cv::Mat features, cv::Mat& scaledFeatures);
  float predict(cv::Mat& data);
  void evalModel(cv::Mat& features, std::vector<int>& labelsVec);
  void (*distanceFunction) (cv::Mat, cv::Mat, cv::Mat&) = NULL;
  
  void readVerificationData();
  void evalVerification();
  //configuration
  std::string      _metric;
  std::string      _pathTrainFeatures;
  std::string      _pathValFeatures;
  std::string      _coeffPath;
  std::string      _biasPath;
  std::string      _scalerMinPath;
  std::string      _scalerDiffPath;
  std::string      _ver1Path;
  std::string      _ver2fPath;
  std::string      _valLabelPath;
  struct Features* _trainFeatures = NULL;
  //data for scaling
  cv::Mat          _maxValue;
  std::string      _pathScaler;
  //learning algorithm
  std::string      _pathComparator;
  std::string      _pathComparatorMat;
  float            _threshold;
  //Sklearn model
  cv::Mat          _coeffSK;
  cv::Mat          _biasSK;
  cv::Mat          _skalerMinSK;
  cv::Mat          _skalerDiffSK;
  //Verification data
  std::vector<int> _idxVer1;
  std::vector<int> _idxVer2;
  std::vector<int> _idxLabels;
  

};


#endif //VERIFICATOR_HPP