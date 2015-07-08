#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/python.hpp>
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
#include "conversion.h"
#include "Serialization.hpp"

namespace py = boost::python;

typedef unsigned char uchar_t;


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

/**
 * Read C++ features from disk and next transform it to one Mat. Then return as Numpy array
 * 
 */
PyObject*
transformFeatures(std::string pathFeatures)
{
  NDArrayConverter cvt;
  Features* features = new Features;
  load(*features, pathFeatures);
  PyObject* ret = cvt.toNDArray(features->data);
  delete features;
  return ret;
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
     py::def("transformFeatures", transformFeatures);
}
