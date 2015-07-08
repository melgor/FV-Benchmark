#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <memory>

struct SVM_Mat
{
  cv::Mat                  weights;
  std::vector<float>         bias;
};

struct Features
{
  cv::Mat                  data;
  std::vector<int>         labels;
  std::vector<std::string> names;
};

BOOST_SERIALIZATION_SPLIT_FREE(::SVM_Mat)
BOOST_SERIALIZATION_SPLIT_FREE(::Features)
BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
namespace boost {
  namespace serialization {
 
    /** Serialization support for SVM_Mat */
    template<class Archive>
    void save(Archive & ar, const ::SVM_Mat& m, const unsigned int version)
    {
      size_t elem_size = m.weights.elemSize();
      size_t elem_type = m.weights.type();
 
      ar & m.weights.cols;
      ar & m.weights.rows;
      ar & elem_size;
      ar & elem_type;
      ar & m.bias;
 
      const size_t data_size = m.weights.cols * m.weights.rows * elem_size;
      ar & boost::serialization::make_array(m.weights.ptr(), data_size);
    }

    /** Serialization support for SVM_Mat */
    template<class Archive>
    void load(Archive & ar, ::SVM_Mat& m, const unsigned int version)
    {
      int cols, rows;
      size_t elem_size, elem_type;
      std::vector<float> bias;
 
      ar & cols;
      ar & rows;
      ar & elem_size;
      ar & elem_type;
      ar & bias;
 
      m.weights.create(rows, cols, elem_type);
      m.bias = bias;
      size_t data_size = m.weights.cols * m.weights.rows * elem_size;
      ar & boost::serialization::make_array(m.weights.ptr(), data_size);
    }

    /** Serialization support for Features */
    template<class Archive>
    void save(Archive & ar, const ::Features& m, const unsigned int version)
    {
      size_t elem_size = m.data.elemSize();
      size_t elem_type = m.data.type();
 
      ar & m.data.cols;
      ar & m.data.rows;
      ar & elem_size;
      ar & elem_type;
      ar & m.labels;
      ar & m.names;
 
      const size_t data_size = m.data.cols * m.data.rows * elem_size;
      ar & boost::serialization::make_array(m.data.ptr(), data_size);
    }

    /** Serialization support for Features */
    template<class Archive>
    void load(Archive & ar, ::Features& m, const unsigned int version)
    {
      int cols, rows;
      std::vector<int> labels;
      std::vector<std::string> names;
      size_t elem_size, elem_type;
 
      ar & cols;
      ar & rows;
      ar & elem_size;
      ar & elem_type;
      ar & labels;
      ar & names;

      m.data.create(rows, cols, elem_type);
      m.labels = labels;
      m.names  = names;
      size_t data_size = m.data.cols * m.data.rows * elem_size;
      ar & boost::serialization::make_array(m.data.ptr(), data_size);
    }

    /** Serialization support for cv::Mat */
    template<class Archive>
    void save(Archive &ar, const cv::Mat &m, const unsigned int __attribute__((unused)) version)
    {
        size_t elem_size = m.elemSize();
        size_t elem_type = m.type();

        ar & m.cols;
        ar & m.rows;
        ar & elem_size;
        ar & elem_type;

        const size_t data_size = m.cols * m.rows * elem_size;
        ar & boost::serialization::make_array(m.ptr(), data_size);
    }

    /** Serialization support for cv::Mat */
    template<class Archive>
    void load(Archive &ar, cv::Mat &m, const unsigned int __attribute__((unused)) version)
    {
        int    cols, rows;
        size_t elem_size, elem_type;

        ar & cols;
        ar & rows;
        ar & elem_size;
        ar & elem_type;

        m.create(rows, cols, elem_type);

        size_t data_size = m.cols * m.rows * elem_size;
        ar & boost::serialization::make_array(m.ptr(), data_size);
    }

//   Try read next object from archive
      template<class Archive, class Stream, class Obj>
      bool try_stream_next(Archive &ar, const Stream &s, Obj &o)
      {
        bool success = false;

        try {
          ar >> o;
          success = true;
        } catch (const boost::archive::archive_exception &e) {
          if (e.code != boost::archive::archive_exception::input_stream_error) {
            throw;
          }
        }
      
        return success;
      }
  }  
}

template<class T>
void compress(T& obj, std::string path)
{
  namespace io = boost::iostreams;
 
 
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  { // use scope to ensure archive and filtering stream buffer go out of scope before stream
    io::filtering_streambuf<io::output> out;
    out.push(io::zlib_compressor(io::zlib::best_speed));
    out.push(ofs);
    boost::archive::binary_oarchive oa(out);
    oa << obj;
    
  }
 
  ofs.close();
};

template<class T>
void load(T& obj, std::string path)
{
  namespace io = boost::iostreams;
 
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  {
    io::filtering_streambuf<io::input> in;
    in.push(io::zlib_decompressor());
    in.push(ifs);
 
    boost::archive::binary_iarchive ia(in);
 
    bool cont = true;
    while (cont)
    {
      cont = boost::serialization::try_stream_next(ia, ifs, obj);
    }
  }
 
  ifs.close();
}

#endif