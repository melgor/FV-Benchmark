/*
* @Author: melgor
* @Date:   2014-05-27 17:08:52
* @Last Modified 2015-04-27
*/

#include <boost/filesystem.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "Utils.hpp"
namespace fs = boost::filesystem;

// Import all images from path, where images are files with endswitch png, jpg, jpeg, JPG
std::vector<std::string>
importImages(std::string path)
{
  std::vector<cv::Mat>  imageColor;
  fs::path p(path);
  std::vector<std::string> image2;
  try
  {
    if(exists(p))    // does p actually exist?
    {
      if(is_directory(p))      // is p a directory?
      {
        fs::directory_iterator end_iter;
        for (fs::directory_iterator dir_itr(p); dir_itr != end_iter; ++dir_itr)
        {

          std::string extension = dir_itr->path().extension().string();
          if(extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".JPG" )
          {
          
	    		image2.push_back ( dir_itr->path().string());
          }
        
        }
      }
      else
      {
        assert(is_directory(p));
      }
    }
    else
    {
      std::cerr<<p << " does not exist. Cretete it ot change directory name"<<std::endl;
      exit(0);
    }
  }

  catch (const fs::filesystem_error& ex)
  {
    std::cerr << ex.what() << '\n';
  }
  
  return image2;
}

//Get fodler name in path. Folder is assumed to have none extension
std::vector<std::string> 
getFolderInPath(std::string path)
{
  std::vector<cv::Mat>  imageColor;
  fs::path p(path);
  std::vector<std::string> image2;
  try
  {
    if(exists(p))    // does p actually exist?
    {
      if(is_directory(p))      // is p a directory?
      {
        fs::directory_iterator end_iter;
        for (fs::directory_iterator dir_itr(p); dir_itr != end_iter; ++dir_itr)
        {

          std::string extension = dir_itr->path().extension().string();
          if(extension == "")
          {
             image2.push_back ( dir_itr->path().string());
          }
        }
      }
      else
      {
        assert(is_directory(p));
      }
    }
    else
    {
      std::cerr<<p << " does not exist. Cretete it ot change directory name"<<std::endl;
      exit(0);
      //assert(exists(p));
    }
  }

  catch (const fs::filesystem_error& ex)
  {
    std::cerr << ex.what() << '\n';
  }
  
  return image2;
}


double
findAngle(
   cv::Point p1
 , cv::Point center
 , cv::Point p2
 )
{
  double p1c  = sqrt( pow(center.x - p1.x, 2) + pow(center.y - p1.y, 2) );
  double p2c  = sqrt( pow(center.x - p2.x, 2) + pow(center.y - p2.y, 2) );
  double p1p2 = sqrt( pow(p2.x     - p1.x, 2) + pow(p2.y     - p1.y, 2) );
  if (p1c == 0 || p2c == 0 || p1p2 ==0)
    return -1000;
  return acos( (p2c * p2c + p1c * p1c - p1p2 * p1p2) / (2 * p2c * p1c)) * 180/CV_PI;
}

double
calcDistance( cv::Point2f p1, cv::Point2f p2)
{
  return cv::norm(p1-p2);
}

bool 
intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2)
{
    cv::Point2f x = o2 - o1;
    cv::Point2f d1 = p1 - o1;
    cv::Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    cv::Point2f r = o1 + d1 * t1;
    std::cerr<<"inter "<< r << std::endl;
    std::cerr<<o1<<p1<<o2<<p2<<std::endl;
    return true;
}

void 
calculateMeanPoint( std::vector<cv::Point2f>& points, cv::Point2f& mean_point)
{
  for(auto& point : points)
  {
    mean_point.x += point.x;
    mean_point.y += point.y;
  }
  mean_point.x /= points.size();
  mean_point.y /= points.size();
}

float 
rectIntersection(cv::Rect& r1, cv::Rect& r2)
{
  cv::Rect rect_over = r1 & r2;
  return rect_over.area()/float(r2.area());
}

