/* 
* @Author: melgor
* @Date:   2015-04-09 10:07:45
* @Last Modified 2015-04-09
* @Last Modified time: 2015-04-09 10:46:32
*/
#include "Distances.hpp"

using namespace std;
using namespace cv;

void
cosineDistance(cv::Mat feat1, cv::Mat feat2, cv::Mat& result)
{
  float dot_prodcut = feat1.dot(feat2);
  float sum_vec1    = sum(feat1.mul(feat1))[0];
  float sum_vec2    = sum(feat2.mul(feat2))[0];

  float result_val  =  dot_prodcut/(sqrt(sum_vec1) * sqrt(sum_vec2));
  std::vector<float> data(1,result_val);
  result            =  Mat( data, true);
}


void
chiSquaredDistance(cv::Mat feat1, cv::Mat feat2, cv::Mat& result)
{
  //dist = (f1 - f2)^2 elementwise
  Mat distance = feat1 - feat2;
  distance     = distance.mul(distance);
//   cerr<<"diff: " << sum(distance)[0] << endl;
  //out        = dist/(f1 + f2) elementwise
  Mat sum_f    = feat1 + feat2;
  //if any value is == 0.0, replace it by 1
//   cerr<<"sum_f: " << sum(sum_f)[0] << endl;
  MatIterator_<float> it_dst = sum_f.begin<float>(), it_end_dst = sum_f.end<float>();
  for(MatIterator_<float> j = it_dst; j != it_end_dst ;++j)
  {
    if (*j == 0.0f)
    {
      *j = 1.0f;
    }
  }
//    cerr<<"sum_f: " << sum(sum_f)[0] << endl;
  //out = dist/(sum) elementwise
  result = distance.mul(1.0/(sum_f)); 
//   std::cerr<<result<<std::endl;
}
