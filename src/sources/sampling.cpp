/*
<%
cfg['compiler_args'] = ['-std=c++11', '-undefined dynamic_lookup']
%>
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>

typedef unsigned int ui;

using namespace std;
namespace py = pybind11;

int randint_(int end)
{
    return rand() % end;
}

py::array_t<int> sample_negative(int user_num, int item_num, int train_num, int neg_num, std::vector<std::vector<int>> allPos){
    // 计算要对每个用户生成的样本数量，即训练样本数量除以用户数量
    int perUserNum = (train_num / user_num); 
    // 计算每个用户样本的个数，包括用户编号、正样本编号和负样本编号
    int row = neg_num + 2; 
    // 创建 [user_num * perUserNum，row] 的 S_array，存储采样结果
    py::array_t<int> S_array = py::array_t<int>({user_num * perUserNum, row});
    // 获取 S_array 的底层数据指针
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;
    // 循环遍历每个用户
    for (int user = 0; user < user_num; user++){
        // 获取该用户正样本列表
        std::vector<int> pos_item = allPos[user];
        // 循环遍历要生成的样本个数
        for (int pair_i = 0; pair_i < perUserNum; pair_i++){
            int negitem = 0;
            // [user * perUserNum + pair_i, 0] 位置写入 user
            ptr[(user * perUserNum + pair_i) * row] = user;
            // [user * perUserNum + pair_i, 1] 位置写入正样本采样
            ptr[(user * perUserNum + pair_i) * row + 1] = pos_item[randint_(pos_item.size())];
            // 负样本采样，采 neg_num 个
            for (int index = 2; index < neg_num + 2; index++){
                do{
                    negitem = randint_(item_num);
                } while (find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
                ptr[(user * perUserNum + pair_i) * row + index] = negitem;
            }
        }
    }
    return S_array;
}

py::array_t<int> sample_negative_ByUser(int item_num, int neg_num, std::vector<int> users, std::vector<std::vector<int>> allPos){
    int col = users.size();
    int row = neg_num + 2;
    py::array_t<int> S_array = py::array_t<int>({col, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user_i = 0; user_i < (int) users.size(); user_i++){
        int user = users[user_i];
        std::vector<int> pos_item = allPos[user];
        int negitem = 0;

        ptr[user_i * row] = user;
        ptr[user_i * row + 1] = pos_item[randint_(pos_item.size())];

        for (int neg_i = 2; neg_i < row; neg_i++){
            do{
                negitem = randint_(item_num);
            } while (
                find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
            ptr[user_i * row + neg_i] = negitem;
        }
    }
    return S_array;
}

void set_seed(unsigned int seed){
    srand(seed);
}

using namespace py::literals;

PYBIND11_MODULE(sampling, m){
//    srand(time(0));
    srand(2020);
    m.doc() = "example plugin";
    m.def("randint", &randint_, "generate int between [0 end]", "end"_a);
    m.def("seed", &set_seed, "set random seed", "seed"_a);
    m.def("sample_negative", &sample_negative, "sampling negatives for all",
          "user_num"_a, "item_num"_a, "train_num"_a, "neg_num"_a, "allPos"_a);
    m.def("sample_negative_ByUser", &sample_negative_ByUser, "sampling negatives for given users",
          "item_num"_a, "neg_num"_a, "users"_a, "allPos"_a);
}