/*
* @Author: melgor
* @Date:   2015-04-09 09:05:37
* @Last Modified 2015-06-19
* @Last Modified time: 2015-06-19 13:45:25
*/

#include "Verificator.hpp"
#include "Distances.hpp"
#include "Serialization.hpp"
#include <boost/algorithm/string.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

Verificator::Verificator(struct Configuration& config)
{
  _pathTrainFeatures = config.trainData;
  _pathValFeatures   = config.valData;
  _threshold         = config.threshold;
  _pathComparator    = config.pathComparator;
  _pathComparatorMat = config.pathComparatorMat;
  _metric            = config.metric;
  _pathScaler        = config.pathScaler;
  _coeffPath         = config.coeffPath;
  _biasPath          = config.biasPath;
  _scalerMinPath     = config.scalerMinPath;
  _scalerDiffPath    = config.scalerDiffPath;
  _ver1Path          = config.ver1Label;
  _ver2fPath         = config.ver2Label;
  _valLabelPath      = config.valLabel;
  if (_metric == "Cosine")
    distanceFunction = &cosineDistance;
  else if(_metric == "Chi")
    distanceFunction = &chiSquaredDistance;

}

void
Verificator::loadModel()
{

  load(_coeffSK, _coeffPath);
  load(_biasSK, _biasPath);
  load(_skalerMinSK, _scalerMinPath);
  load(_skalerDiffSK, _scalerDiffPath);

  cerr<<"Model Loaded"<<endl;
}

void readData( string& path, vector<string>& names, vector<int>& labels)
{
  string line;
  ifstream myfile (path);
  vector<string> splitteds;
  while ( getline (myfile, line) )
  {
    boost::split(splitteds, line, boost::is_any_of(" "));
    labels.push_back(std::stoi(splitteds[1]));
    names.push_back(splitteds[0]);
    splitteds.clear();
   
  }
  myfile.close();
}

void 
Verificator::readVerificationData()
{
  vector<string> names_ver_1, names_ver_2, names_val;
  vector<int>    labels_ver_1, labels_ver_2, labels_val;
  
  readData(_ver1Path, names_ver_1, labels_ver_1);
  readData(_ver2fPath, names_ver_2, labels_ver_2);
  readData(_valLabelPath, names_val, labels_val);
  cerr<<"Sizes: "<< labels_ver_1.size() << " "<< labels_ver_2.size() <<" " << labels_val.size() << endl;
  
  //find index of images from feature vector
  _idxVer1.reserve( names_ver_1.size());
  _idxVer2.reserve( names_ver_1.size());
  for(uint i = 0; i < names_ver_1.size(); i++)
  {
    string ver_1 = names_ver_1[i];
    string ver_2 = names_ver_2[i];
    int idx_ver_1 = std::distance(names_val.begin(), std::find(names_val.begin(), names_val.end(), ver_1));
    int idx_ver_2 = std::distance(names_val.begin(), std::find(names_val.begin(), names_val.end(), ver_2));
    _idxVer1.push_back(idx_ver_1);
    _idxVer2.push_back(idx_ver_2);
    if( labels_ver_1[i] == labels_ver_2[i])
    {
      _idxLabels.push_back(1);
    }
    else
    {
      _idxLabels.push_back(0);
    }
    if (i%1000  == 0)
      cerr<<"Set " << i << endl;
  }
}

void 
Verificator::evalVerification()
{
  
  Features* train_Features      = new Features;
  load( *train_Features, _pathValFeatures);
  Mat scaled_features;
  scaleData(train_Features->data, scaled_features);
  float thres = -0.0610386574332;
  int good = 0;
  for(uint i = 0; i < _idxVer1.size(); i++)
  {
    float score = predValue(scaled_features.row(_idxVer1[i]), scaled_features.row(_idxVer2[i]));
//     cerr<<"Score: "<< score << " label " << _idxLabels[i] << endl;
    int label = 0;
    if(score > thres)
      label = 1;
    
    if( label == _idxLabels[i])
      good++;

  }
  
  cerr<<"ACC: "<< (float(good)/_idxVer1.size()) << endl;

  delete train_Features;
  
}

void 
Verificator::verify()
{ 
  loadModel();
  //load validation data
  _trainFeatures      = new Features;
  load( *_trainFeatures, _pathValFeatures);

  //scale all features
  Mat scale_data_all;
  for(int row = 0; row < _trainFeatures->data.rows; row++)
  {
    Mat scaled_data;
    scaleData(_trainFeatures->data.row(row), scaled_data );
    scale_data_all.push_back(scaled_data);
  }
  // Mat f1 = scale_data_all.row(0), f2 = scale_data_all.row(0);
  // cerr<<"Compare: "<<compare(f1,f2) << endl;
  Mat features;
  vector<int> labelsVec;
  cerr<<"Create Verification Task"<<endl;
  prepareVerificationData(scale_data_all, features, labelsVec);
  evalModel(features, labelsVec);
  // _comparatorLinear->loadData(features, labelsVec);
  // _comparatorLinear->evalModel();

}

float 
Verificator::predict(Mat& data)
{
  float prob_class = _coeffSK.dot(data) +  _biasSK.at<double>(0,0);
  return prob_class;
}

void 
Verificator::evalModel(  
                          Mat& features
                        , vector<int>& labelsVec
                        )
{
  double npos, nneg,all;
  npos = nneg = all = 0;

  double cpos, cneg;
  cpos = cneg = 0;
  int label = 0;

  for(int data  = 0; data < features.rows; data++)
  {
    Mat feat = features.row(data);
    float pred  = predict(feat);
    cerr<<"Pred: " << pred << " Label: "<<  labelsVec[data] << endl;
    label = 0;
    if(pred > _threshold)
      label = 1;

    if (label == labelsVec[data] && labelsVec[data] == int(1.0))
    {
      ++cpos;
      ++npos;
      // cerr<<"0"<<endl;
    }
    else if (labelsVec[data] == int(1.0))
    {
      ++npos;
      // cerr<<"1"<<endl;
    }
    else if(label == labelsVec[data] && labelsVec[data] == int(0.0))
    {
      ++cneg;
      ++nneg;
      // cerr<<"2"<<endl;
    }
    else if (labelsVec[data] == int(0.0))
    {
      ++nneg;
      // cerr<<"3"<<endl;
    }
    
    if(label == labelsVec[data])
    {
      all++;
    }

  }
 cerr<<"Acc: True: " <<cpos / npos<<" False: "<< cneg / nneg<< " all: "<< all/float(labelsVec.size())<< endl;


}

//compare if two features descrive same person or not
int
Verificator::compare(
                      Mat& featureOne
                    , Mat& featureTwo
                    )
{
  //compute feature representation
  Mat feat;
  (*distanceFunction)(featureOne,featureTwo,feat);
  
  //predict value or apply threshold
  return predict(feat);//_comparatorLinear->predict_class(feat);
}

float 
Verificator::predValue(cv::Mat featureOne, cv::Mat featureTwo)
{
   Mat feat;
  (*distanceFunction)(featureOne,featureTwo,feat);
  return predict(feat);
}

//scale data
void 
Verificator::scaleData(
                        Mat  features
                      , Mat& scaledFeatures
                      )
{
  scaledFeatures = features.clone();
  for (int r = 0; r < features.rows; ++r) 
  {
    scaledFeatures.row(r) =  ((features.row(r) - _skalerMinSK).mul(1.0/_skalerDiffSK));
  }
  
}

void 
Verificator::prepareVerificationData(
                              Mat& scaledFetures
                            , Mat& featuresVer
                            , vector<int>& labelsVecVer
                            )
{
  int num_example =  _trainFeatures->labels.size();
  labelsVecVer.resize(2 * num_example,0);
  RNG rng;
  int feat_1 = 0, feat_2 = 0;
  //produce only positive example
  for(int i = 0; i < 1.0* num_example; i++)
  {
    feat_1 = 0;
    feat_2 = 10;
    //cerr<<i<<" "<<(feat_1 == feat_2) << " "<< !feat_1<<endl;
    while(_trainFeatures->labels[feat_2] != _trainFeatures->labels[feat_1])
    {
      //get two random example and create vector for feature based on distance
      feat_1 = rng.uniform(int(0), num_example -1);
      feat_2 = rng.uniform(int(0), num_example -1);
       //cerr<<i<<" "<<(feat_1 == feat_2) << " "<< !feat_1<<endl;
    }

    // cerr<<"Labels: "<<_trainFeatures->labels[feat_1] <<" "<< _trainFeatures->labels[feat_2]<<" "  << feat_1 << "  "<<feat_2<<" names: "<<_trainFeatures->names[feat_1]<<" "<<_trainFeatures->names[feat_2] << endl;
    Mat feat;
    (*distanceFunction)(scaledFetures.row(feat_1),scaledFetures.row(feat_2),feat);
    featuresVer.push_back(feat);
    // if (_trainFeatures->labels[feat_1] == _trainFeatures->labels[feat_2])
    labelsVecVer[i] = 1;
  }

  //produce only negative example
  for(int i = 1.0* num_example; i < 2*num_example; i++)
  {
    feat_1 = 0;
    feat_2 = 0;
    while(_trainFeatures->labels[feat_2] == _trainFeatures->labels[feat_1])
    {
      //get two random example and create vector for Chi distance
      feat_1 = rng.uniform(int(0), num_example -1);
      feat_2 = rng.uniform(int(0), num_example -1);
    }

    Mat feat;
    (*distanceFunction)(scaledFetures.row(feat_1),scaledFetures.row(feat_2),feat);
    featuresVer.push_back(feat);

  }
}


Verificator::~Verificator()
{

  // delete _comparatorLinear;
  if (_trainFeatures != NULL)
      delete _trainFeatures;

}

