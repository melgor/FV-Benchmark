/*
* @Author: melgor
* @Date:   2014-05-26 22:22:02
* @Last Modified 2015-06-19
*/
#include <chrono>
#include <iostream>
#include <boost/filesystem.hpp>
#include <string>
#include "model/Serialization.hpp"
#include "model/Distances.hpp"
#include "model/Verificator.hpp"
using namespace std;
using namespace cv;





int main(int argc, char **argv)
{
  struct Configuration conf;
  conf.read(argc, argv);
  Verificator ver(conf);
  cerr<<"LOAD"<<endl;
  ver.loadModel();
  ver.readVerificationData();
  ver.evalVerification();
  
//   string mystring = string(argv[1]);
//   Features* features = new Features;
//   load(*features, mystring);
// //   cerr<<features->data.row(0) << endl;
//   Mat res;
//   chiSquaredDistance(features->data.row(0), features->data.row(1280), res);
// // //   res = features->data.row(0)- features->data.row(1);
//   cerr<<"Chi: "<< sum(res)[0]<<endl;
//   cosineDistance(features->data.row(0), features->data.row(1280), res);
//   cerr<<"cosine: "<< res<<endl;
//   delete features;
  return 0;
}
