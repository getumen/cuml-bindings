#include <fstream>

#include <treelite/c_api.h>
#include <gtest/gtest.h>

#include "cuml4c/device_resource_handle.h"
#include "cuml4c/memory_resource.h"
#include "cuml4c/fil.h"

TEST(FILTest, TestTreelite)
{

    ModelHandle handle;
    auto res = TreeliteLoadXGBoostModel("testdata/xgboost.model", &handle);
    EXPECT_EQ(res, 0);

    size_t num_classes = 0;
    res = TreeliteQueryNumClass(handle, &num_classes);
    EXPECT_EQ(res, 0);

    size_t num_features = 0;
    res = TreeliteQueryNumFeature(handle, &num_features);
    EXPECT_EQ(res, 0);

    res = TreeliteFreeModel(handle);
    EXPECT_EQ(res, 0);
}

TEST(FILTest, TestFIL)
{
    DeviceResourceHandle device_resource_handle;
    CreateDeviceResourceHandle(&device_resource_handle);

    DeviceMemoryResource mr;
    UseArenaMemoryResource(&mr);

    FILModelHandle handle;
    auto res = FILLoadModel(device_resource_handle, 0, "testdata/xgboost.model", 0, true, 0.5, 0, 0, 1, 0, &handle);
    EXPECT_EQ(res, 0);

    size_t num_classes = 0;
    res = FILGetNumClasses(handle, &num_classes);
    EXPECT_EQ(res, 0);

    std::vector<float> feature;
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

    std::vector<float> preds(num_row * num_classes);
    EXPECT_EQ(num_row, 114);
    EXPECT_EQ(feature.size(), 114 * 30);

    res = FILPredict(device_resource_handle, handle, feature.data(), num_row, true, preds.data());
    EXPECT_EQ(res, 0);

    {
        std::ifstream ifs_csv_file("testdata/score-xgboost.csv");
        std::string line;
        int index = 0;
        while (std::getline(ifs_csv_file, line))
        {
            auto expected = std::stof(line);
            EXPECT_NEAR(expected, preds[2 * index + 1], 0.0001);
            index++;
        }
    }

    res = FILFreeModel(device_resource_handle, handle);
    EXPECT_EQ(res, 0);

    ResetMemoryResource(mr, 2);

    FreeDeviceResourceHandle(device_resource_handle);
}
