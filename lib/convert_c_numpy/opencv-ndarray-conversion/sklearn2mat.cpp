/*
Convert Models learned in sklearn (python) to C++ cv::Mat
*/

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/python.hpp>
#include "Serialization.hpp"
#include "conversion.h"

namespace py = boost::python;

typedef unsigned char uchar_t;




void
saveSVMModel(PyObject *coeff,PyObject *thres,PyObject *scalerMin, PyObject *scalerDiff)
{
  NDArrayConverter cvt;
  cv::Mat coeff_cv { cvt.toMat(coeff) };
  cv::Mat thres_cv { cvt.toMat(thres) };
  cv::Mat scalerMin_cv { cvt.toMat(scalerMin) };
  cv::Mat scalerDiff_cv { cvt.toMat(scalerDiff) };
  std::cerr<<thres_cv << std::endl;
}

static void init()
{
    Py_Initialize();
    import_array();
}

BOOST_PYTHON_MODULE(examples)
{
    init();
    py::def("saveSVMModel", saveSVMModel);
}
