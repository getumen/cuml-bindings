#include <fstream>

#include <gtest/gtest.h>

#include "cuml4c/device_resource_handle.h"
#include "cuml4c/linear_regression.h"

TEST(GLMTest, TestLinearRegression)
{
    DeviceResourceHandle device_resource_handle;
    CreateDeviceResourceHandle(&device_resource_handle);

    std::vector<float> feature;
    size_t num_col = 30;
    size_t num_row = 0;

    {
        std::ifstream ifs_csv_file("testdata/feature.csv");
        std::string line;
        while (std::getline(ifs_csv_file, line))
        {
            std::stringstream ss(line);
            std::string val;
            num_row++;
            while (std::getline(ss, val, ','))
            {
                feature.push_back(std::stof(val));
            }
        }
    }

    std::vector<float> labels;
    {
        std::ifstream ifs_csv_file("testdata/label.csv");
        std::string line;
        while (std::getline(ifs_csv_file, line))
        {
            labels.push_back(std::stof(line));
        }
    }

    std::vector<float> coef(num_col);
    std::vector<float> intercept(1);

    {
        auto res = OlsFit(
            device_resource_handle,
            feature.data(),
            num_row,
            num_col,
            labels.data(),
            true,
            true,
            0,
            coef.data(),
            intercept.data());

        EXPECT_EQ(res, 0);
    }

    std::vector<float> preds(num_row);
    {
        auto res = GemmPredict(
            device_resource_handle,
            feature.data(),
            num_row,
            num_col,
            coef.data(),
            intercept[0],
            preds.data());
    }

    FreeDeviceResourceHandle(device_resource_handle);
}

TEST(GLMTest, TestRidgeRegression)
{
    DeviceResourceHandle device_resource_handle;
    CreateDeviceResourceHandle(&device_resource_handle);

    std::vector<float> feature;
    size_t num_col = 30;
    size_t num_row = 0;

    {
        std::ifstream ifs_csv_file("testdata/feature.csv");
        std::string line;
        while (std::getline(ifs_csv_file, line))
        {
            std::stringstream ss(line);
            std::string val;
            num_row++;
            while (std::getline(ss, val, ','))
            {
                feature.push_back(std::stof(val));
            }
        }
    }

    std::vector<float> labels;
    {
        std::ifstream ifs_csv_file("testdata/label.csv");
        std::string line;
        while (std::getline(ifs_csv_file, line))
        {
            labels.push_back(std::stof(line));
        }
    }

    std::vector<float> alpha(1, 1.0f);

    std::vector<float> coef(num_col);
    std::vector<float> intercept(1);

    {
        auto res = RidgeFit(
            device_resource_handle,
            feature.data(),
            num_row,
            num_col,
            labels.data(),
            alpha.data(),
            alpha.size(),
            true,
            true,
            0,
            coef.data(),
            intercept.data());

        EXPECT_EQ(res, 0);
    }

    std::vector<float> preds(num_row);
    {
        auto res = GemmPredict(
            device_resource_handle,
            feature.data(),
            num_row,
            num_col,
            coef.data(),
            intercept[0],
            preds.data());
    }

    FreeDeviceResourceHandle(device_resource_handle);
}
