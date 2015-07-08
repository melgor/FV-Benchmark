#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/python.hpp>
#include "conversion.h"
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
namespace py = boost::python;

typedef unsigned char uchar_t;



BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
namespace boost {
  namespace serialization {

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

/**
 * Displays an image, passed in from python as an ndarray.
 */
void
display(PyObject *img)
{
    NDArrayConverter cvt;
    cv::Mat mat { cvt.toMat(img) };
    
    cv::namedWindow("display", CV_WINDOW_NORMAL);
    cv::imshow("display", mat);
    cv::waitKey(0);
}


/**
 * Converts a grayscale image to a bilevel image.
 */
PyObject*
binarize(PyObject *grayImg, short threshold)
{
    NDArrayConverter cvt;
    cv::Mat img { cvt.toMat(grayImg) };
    for (int i = 0; i < img.rows; ++i)
    {
        uchar_t *ptr = img.ptr<uchar_t>(i);
        for (int j = 0; j < img.cols; ++j)
        {
            ptr[j] = ptr[j] < threshold ? 0 : 255;
        }
    }
    return cvt.toNDArray(img);
}

/**
 * Multiplies two ndarrays by first converting them to cv::Mat and returns
 * an ndarray containing the result back.
 */
PyObject*
mul(PyObject *left, PyObject *right)
{
    NDArrayConverter cvt;
    cv::Mat leftMat, rightMat;
    leftMat = cvt.toMat(left);
    rightMat = cvt.toMat(right);
    auto r1 = leftMat.rows, c1 = leftMat.cols, r2 = rightMat.rows,
         c2 = rightMat.cols;
    // Work only with 2-D matrices that can be legally multiplied.
    if (c1 != r2)
    {
        PyErr_SetString(PyExc_TypeError, 
                        "Incompatible sizes for matrix multiplication.");
        py::throw_error_already_set();
    }
    cv::Mat result = leftMat * rightMat;

    PyObject* ret = cvt.toNDArray(result);

    return ret;
}

void
saveSVMModel(PyObject *coeff,PyObject *thres,PyObject *scalerMin, PyObject *scalerDiff)
{
  NDArrayConverter cvt;
  cv::Mat coeff_cv { cvt.toMat(coeff) };
  cv::Mat thres_cv { cvt.toMat(thres) };
  cv::Mat scalerMin_cv { cvt.toMat(scalerMin) };
  cv::Mat scalerDiff_cv { cvt.toMat(scalerDiff) };
  scalerMin_cv = scalerMin_cv.t();
  scalerDiff_cv = scalerDiff_cv.t();
  coeff_cv      = coeff_cv.t();
  coeff_cv.convertTo(coeff_cv, CV_32F);
  std::cerr<<"BIAS: "<< thres_cv << std::endl;
  compress(coeff_cv, "coeff_cv.bin");
  compress(thres_cv, "thres_cv.bin");
  compress(scalerMin_cv, "scalerMin_cv.bin");
  compress(scalerDiff_cv, "scalerDiff_cv.bin");
  
}

static void init()
{
    Py_Initialize();
    import_array();
}

BOOST_PYTHON_MODULE(examples)
{
    init();
    py::def("display", display);
    py::def("binarize", binarize);
    py::def("mul", mul);
    py::def("saveSVMModel", saveSVMModel);
}
