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

namespace pse_adaptor {
    void get_kernals(const int *data, vector<long int> data_shape, vector<Mat> &kernals) {
        for (int i = 0; i < data_shape[0]; ++i) {
            Mat kernal = Mat::zeros(data_shape[1], data_shape[2], CV_8UC1);
            for (int x = 0; x < kernal.rows; ++x) {
                for (int y = 0; y < kernal.cols; ++y) {
                    kernal.at<char>(x, y) = data[i * data_shape[1] * data_shape[2] + x * data_shape[2] + y];
                }
            }
            kernals.emplace_back(kernal);
        }
    }

    void growing_text_line(vector<Mat> &kernals, vector<vector<int>> &text_line, float min_area) {
        
        Mat label_mat;
        int label_num = connectedComponents(kernals[kernals.size() - 1], label_mat, 4);

        // cout << "label num: " << label_num << endl;
        
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
            vector<int> row(label_mat.cols);
            for (int y = 0; y < label_mat.cols; ++y) {
                int label = label_mat.at<int>(x, y);
                
                if (label == 0) continue;
                if (area[label] < min_area) continue;
                
                Point point(x, y);
                queue.push(point);
                row[y] = label;
            }
            text_line.emplace_back(row);
        }

        // cout << "ok" << endl;
        
        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};

        for (int kernal_id = kernals.size() - 2; kernal_id >= 0; --kernal_id) {
            while (!queue.empty()) {
                Point point = queue.front(); queue.pop();
                int x = point.x;
                int y = point.y;
                int label = text_line[x][y];
                // cout << text_line.size() << ' ' << text_line[0].size() << ' ' << x << ' ' << y << endl;

                bool is_edge = true;
                for (int d = 0; d < 4; ++d) {
                    int tmp_x = x + dx[d];
                    int tmp_y = y + dy[d];

                    if (tmp_x < 0 || tmp_x >= (int)text_line.size()) continue;
                    if (tmp_y < 0 || tmp_y >= (int)text_line[1].size()) continue;
                    if (kernals[kernal_id].at<char>(tmp_x, tmp_y) == 0) continue;
                    if (text_line[tmp_x][tmp_y] > 0) continue;

                    Point point(tmp_x, tmp_y);
                    queue.push(point);
                    text_line[tmp_x][tmp_y] = label;
                    is_edge = false;
                }

                if (is_edge) {
                    next_queue.push(point);
                }
            }
            swap(queue, next_queue);
        }
    } 

    vector<vector<int>> pse(py::array_t<int, py::array::c_style | py::array::forcecast> quad_n9, float min_area) {
        auto buf = quad_n9.request();
        auto data = static_cast<int *>(buf.ptr);
        vector<Mat> kernals;
        get_kernals(data, buf.shape, kernals);

        // cout << "min_area: " << min_area << endl;
        // for (int i = 0; i < kernals.size(); ++i) {
        //     cout << "kernal" << i <<" shape: " << kernals[i].rows << ' ' << kernals[i].cols << endl;
        // }
        
        vector<vector<int>> text_line;
        growing_text_line(kernals, text_line, min_area);

        return text_line;
    }
}

PYBIND11_PLUGIN(adaptor) {
    py::module m("adaptor", "pse");

    m.def("pse", &pse_adaptor::pse, "pse");

    return m.ptr();
}