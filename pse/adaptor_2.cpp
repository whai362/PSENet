#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include <iostream>
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace py = pybind11;

namespace lanms_adaptor {
	vector<Mat> get_kernals(const int *data, vector<long int> data_shape) {
		vector<Mat> kernals;
		for (int i = 0; i < data_shape[0]; ++i) {
			Mat kernal = Mat::zeros(data_shape[1], data_shape[2], CV_8UC1);
			for (int x = 0; x < kernal.rows; ++x) {
				for (int y = 0; y < kernal.cols; ++y) {
					kernal.at<char>(x, y) = data[i * data_shape[1] * data_shape[2] + x * data_shape[2] + y];
				}
			}
			kernals.emplace_back(kernal);
		}
		return kernals;
	}

    Mat growing_text_line(vector<Mat> kernals) {
        int th1 = 10;
        // int th1 = 0;
        Mat text_line = Mat::zeros(kernals[0].size(), CV_32SC1);
        
        Mat label_mat;
        int label_num = connectedComponents(kernals[kernals.size() - 1], label_mat, 4);
        
        int area[label_num + 1];
        memset(area, 0, sizeof(area));
        for (int x = 0; x < label_mat.rows; ++x) {
            for (int y = 0; y < label_mat.cols; ++y) {
                int label = label_mat.at<int>(x, y);
                if (label == 0) continue;
                area[label] += 1;
            }
        }
        queue<Point> queue, next_queue;
        for (int x = 0; x < label_mat.rows; ++x) {
            for (int y = 0; y < label_mat.cols; ++y) {
                int label = label_mat.at<int>(x, y);
                if (label == 0) continue;
                if (area[label] < th1) continue;
                Point point(x, y);
                queue.push(point);
                text_line.at<int>(x, y) = label;
            }
        }

        // cout << text_line << endl;
        
        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};
        
        for (int kernal_id = kernals.size() - 2; kernal_id >= 0; --kernal_id) {
            while (!queue.empty()) {
                Point point = queue.front(); queue.pop();
                int x = point.x;
                int y = point.y;
                int label = text_line.at<int>(x, y);

                bool is_edge = true;
                for (int d = 0; d < 4; ++d) {
                    int tmp_x = x + dx[d];
                    int tmp_y = y + dy[d];

                    if (tmp_x < 0 || tmp_x >= text_line.rows) continue;
                    if (tmp_y < 0 || tmp_y >= text_line.cols) continue;
                    if (kernals[kernal_id].at<char>(tmp_x, tmp_y) == 0) continue;
                    if (text_line.at<int>(tmp_x, tmp_y) > 0) continue;

                    Point point(tmp_x, tmp_y);
                    queue.push(point);
                    text_line.at<int>(tmp_x, tmp_y) = label;
                    is_edge = false;
                }

                if (is_edge) {
                    next_queue.push(point);
                }
            }

            /*
            label_num = connectedComponents(kernals[kernal_id], label_mat, 4);

            int area[label_num + 1];
            memset(area, 0, sizeof(area));
            for (int x = 0; x < label_mat.rows; ++x) {
                for (int y = 0; y < label_mat.cols; ++y) {
                    int label = label_mat.at<int>(x, y);
                    if (label == 0) continue;
                    area[label] += 1;
                }
            }

            for (int x = 0; x < label_mat.rows; ++x) {
                for (int y = 0; y < label_mat.cols; ++y) {
                    int label = label_mat.at<int>(x, y);
                    if (label == 0) continue;
                    if (area[label] < th1) continue;
                    if (text_line.at<int>(x, y) > 0) continue;
                    text_line.at<int>(x, y) = label + bias;
                }
            }
            bias += label_num;
            */

            /*
            for (int x = 0; x < text_line.rows; ++x) {
                for (int y = 0; y < text_line.cols; ++y) {
                    if (text_line.at<int>(x, y) == 0) continue;
                    Point point(x, y);
                    queue.push(point);
                }
            }
            */

            swap(queue, next_queue);
        }

        // cout << text_line << endl;

        return text_line;
    } 

	vector<vector<int>> merge_quadrangle_n9(py::array_t<int, py::array::c_style | py::array::forcecast> quad_n9) {
		auto buf = quad_n9.request();
		auto data = static_cast<int *>(buf.ptr);
        vector<Mat> kernals = get_kernals(data, buf.shape);
        
        Mat _text_line = growing_text_line(kernals);

        // cout << _text_line << endl;
        vector<vector<int>> text_line;
        for (int x = 0; x < _text_line.rows; ++x) {
            vector<int> row;
            for (int y = 0; y < _text_line.cols; ++y) {
                row.emplace_back(_text_line.at<int>(x, y));
            }
            text_line.emplace_back(row);
        }

        return text_line;
	}
}

PYBIND11_PLUGIN(adaptor) {
	py::module m("adaptor", "NMS");

	m.def("merge_quadrangle_n9", &lanms_adaptor::merge_quadrangle_n9, "merge quadrangels");

	return m.ptr();
}

