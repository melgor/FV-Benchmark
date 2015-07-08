#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <fstream>
#include <memory>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "Parser.hpp"

struct Configuration
{
  bool         reset;
  //mode of program
  std::string  mode;
  std::string  mainFolder;
  //input
  std::string  nameScene;
  std::string  folderpath;
  //Face Detection
  std::string  posemodel;
  std::string  facemodel;
  float        padDetection;
  float        resizeImageRatio;
  std::string  calibOption;
  bool         symetry;
  std::string  model2D_6;
  std::string  model2D_68;
  std::string  frontalization;
  //Net
  std::string  prototxt;
  std::string  caffemodel;
  bool         gpu;
  int          gpuID;
  std::string  layer;
  //Extractor
  std::string  extractorFolder;
  std::string  extractorImageList;
  //Verification
  std::string  trainData;
  std::string  valData;
  std::string  metric;
  std::string  pathComparator;
  std::string  pathComparatorMat;
  std::string  pathScaler;
  std::string  faceData;
  std::string  faceLabels;
  std::string  valLabel;
  std::string  ver1Label;
  std::string  ver2Label;
  float        threshold;
  bool         scaleFeature;
  //Verification-Sklearn model
  std::string  coeffPath;
  std::string  biasPath;
  std::string  scalerMinPath;
  std::string  scalerDiffPath;
  //daemon
  std::string watchFolder; 
  std::string pathLog;




  void read(int argc, char** argv)
  {
    Parser parser;
    parser.read(argc, argv);

    reset          = parser.reset;
    nameScene      = parser.scene;
    folderpath     = parser.folderpath;

    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(parser.config, pt);
    //mode
    mode               = pt.get<std::string>("Mode.Mode");
    mainFolder         = pt.get<std::string>("Mode.Folder");
    //face detection
    posemodel          = mainFolder + pt.get<std::string>("FaceDecetion.PoseModel");
    facemodel          = mainFolder + pt.get<std::string>("FaceDecetion.FaceModel");
    padDetection       = pt.get<float>("FaceDecetion.PadDetection");
    resizeImageRatio   = pt.get<float>("FaceDecetion.ResizeImageRatio");
    calibOption        = pt.get<std::string>("FaceDecetion.CalibOption");
    symetry            = pt.get<bool>("FaceDecetion.Symetry");
    model2D_6          = mainFolder + pt.get<std::string>("FaceDecetion.Model2D_6points");
    model2D_68         = mainFolder + pt.get<std::string>("FaceDecetion.Model2D_68points");
    frontalization     = pt.get<std::string>("FaceDecetion.Frontalization");
    //net 
    prototxt           = mainFolder + pt.get<std::string>("Net.Prototxt");
    caffemodel         = mainFolder + pt.get<std::string>("Net.CaffeModel");
    layer              = pt.get<std::string>("Net.Layer");
    gpu                = pt.get<bool>("Net.GPU");
    gpuID              = pt.get<int>("Net.GPU_ID");
    //Extractor
    extractorFolder    = mainFolder + pt.get<std::string>("Extract.Folder");
    extractorImageList = mainFolder + pt.get<std::string>("Extract.ImageListDB");
    //Verificator
    trainData          = mainFolder + pt.get<std::string>("Verification.TrainData");
    valData            = mainFolder + pt.get<std::string>("Verification.ValData");
    metric             = pt.get<std::string>("Verification.Metric");
    pathComparator     = mainFolder + pt.get<std::string>("Verification.ComparatorPath");
    pathComparatorMat  = mainFolder + pt.get<std::string>("Verification.ComparatorPathMat");
    threshold          = pt.get<float>("Verification.Thres");
    pathScaler         = mainFolder + pt.get<std::string>("Verification.ScalerPath");
    scaleFeature       = pt.get<bool>("Verification.ScaleFeature");
    faceData           = mainFolder + pt.get<std::string>("Verification.FaceData");
    faceLabels         = mainFolder + pt.get<std::string>("Verification.FaceLabels");
    valLabel           = pt.get<std::string>("TestModel.val_path");
    ver1Label          = pt.get<std::string>("TestModel.val_ver1");
    ver2Label          = pt.get<std::string>("TestModel.val_ver2");
    //Verification-Sklearn model
    coeffPath          = mainFolder + pt.get<std::string>("Verification.CoeffPath");
    biasPath           = mainFolder + pt.get<std::string>("Verification.BiasPath");
    scalerMinPath      = mainFolder + pt.get<std::string>("Verification.ScalerMinPath");
    scalerDiffPath     = mainFolder + pt.get<std::string>("Verification.ScalerDiffPath");
    //Daemon
    watchFolder        = mainFolder + pt.get<std::string>("Daemon.WatchFolder");
    pathLog            = mainFolder + pt.get<std::string>("Daemon.LogFolder");
  }

  void print()
  {
    LOG(WARNING)<<"-------------------------------------";
    LOG(WARNING)<<"------------Configuration----------------";
    LOG(WARNING)<<"NameScene:  "  <<nameScene;
    LOG(WARNING)<<"Folderpath: "  <<folderpath;
    LOG(WARNING)<<"Mode:       "  << mode;
    LOG(WARNING)<<"Main:       "  << mainFolder;
    LOG(WARNING)<<"------------Face Detection----------------";
    LOG(WARNING)<<"Posemodel:        "<<posemodel;
    LOG(WARNING)<<"Facemodel:        "<<facemodel;
    LOG(WARNING)<<"PadDetection:     "<<padDetection;
    LOG(WARNING)<<"ResizeImageRatio: "<<resizeImageRatio;
    LOG(WARNING)<<"CalibOption:      "<<calibOption;
    LOG(WARNING)<<"Symetry:          "<<symetry;
    LOG(WARNING)<<"Model2D_6:        "<<model2D_6;
    LOG(WARNING)<<"Model2D_68:       "<<model2D_68;
    LOG(WARNING)<<"------------Net---------------------------";
    LOG(WARNING)<<"Prototxt:       "<<prototxt;
    LOG(WARNING)<<"Caffemodel:     "<<caffemodel;
    LOG(WARNING)<<"Layer:          "<<layer;
    LOG(WARNING)<<"Gpu:            "<<gpu;
    LOG(WARNING)<<"GpuID:          "<<gpuID;
    LOG(WARNING) <<"-------------Extractor---------";
    LOG(WARNING) <<"RxtractorFolder:    "<< extractorFolder;
    LOG(WARNING) <<"RxtractorImageList: "<< extractorImageList;
    LOG(WARNING) <<"-------------Verification---------";
    LOG(WARNING) <<"TrainData:         "<<trainData;
    LOG(WARNING) <<"ValData:           "<<valData;
    LOG(WARNING) <<"FaceData:          "<<faceData;
    LOG(WARNING) <<"FaceLabels:        "<<faceLabels;
    LOG(WARNING) <<"Metric:            "<<metric;
    LOG(WARNING) <<"Thres:             "<<threshold;
    LOG(WARNING) <<"ScaleFeature:      "<<scaleFeature;
    LOG(WARNING) <<"ComparatorPath:    "<<pathComparator;
    LOG(WARNING) <<"ComparatorPathMat: "<<pathComparatorMat;
    LOG(WARNING) <<"ScalerPath:        "<<pathScaler;
    LOG(WARNING) <<"-------------Daemon---------";
    LOG(WARNING) <<"WatchFolder:     "<<watchFolder;
    LOG(WARNING) <<"LogFolder:       "<<pathLog;
    LOG(WARNING)<<"-------------------------------------";
  };

};

std::vector<std::string> importImages(std::string path);
std::vector<std::string> getFolderInPath(std::string path);
double findAngle( cv::Point p1, cv::Point center, cv::Point p2);
double calcDistance( cv::Point2f p1, cv::Point2f p2);
// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2);

void calculateMeanPoint(std::vector<cv::Point2f>& points, cv::Point2f& mean_point);

float rectIntersection(cv::Rect& r1, cv::Rect& r2);

template<class T>
void savePoints(std::string& name, std::vector<T>& points)
{
   cv::FileStorage fs(name, cv::FileStorage::WRITE);
   write( fs , "points", points );
   fs.release();
}

template<class T>
void  loadPoints(std::string& name, std::vector<T>& points)
{
  cv::FileStorage fs2(name, cv::FileStorage::READ);
  cv::FileNode kptFileNode = fs2["points"];
  read( kptFileNode, points );
  fs2.release();
}

#endif