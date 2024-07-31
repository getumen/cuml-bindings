#include <fstream>

#include <gtest/gtest.h>

#include "cuml4c/device_resource_handle.h"
#include "cuml4c/memory_resource.h"
#include "cuml4c/agglomerative_clustering.h"
#include "cuml4c/dbscan.h"
#include "cuml4c/kmeans.h"

TEST(ClusteringTest, TestAgglomerativeClustering)
{
    DeviceResourceHandle device_resource_handle;
    CreateDeviceResourceHandle(&device_resource_handle);

    DeviceMemoryResource mr;
    UseArenaMemoryResource(&mr, 1024 * 1024);

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
    size_t num_col = 30;

    int n_clusters = 0;
    std::vector<int> labels(num_row);
    std::vector<int> children((num_row - 1) * 2);
    {
        auto res = AgglomerativeClusteringFit(
            device_resource_handle,
            feature.data(),
            num_row,
            num_col,
            false,
            0,
            10,
            5,
            &n_clusters,
            labels.data(),
            children.data());
    }

    ResetMemoryResource(mr, 2);

    FreeDeviceResourceHandle(device_resource_handle);
}

TEST(ClusteringTest, TestDBScan)
{
    DeviceResourceHandle device_resource_handle;
    CreateDeviceResourceHandle(&device_resource_handle);

    DeviceMemoryResource mr;
    UseArenaMemoryResource(&mr, 1024 * 1024);

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
    size_t num_col = 30;

    std::vector<int> labels(num_row);
    {
        auto res = DbscanFit(
            device_resource_handle,
            feature.data(),
            num_row,
            num_col,
            5,
            3.0,
            5,
            0,
            4,
            labels.data());
    }

    ResetMemoryResource(mr, 2);

    FreeDeviceResourceHandle(device_resource_handle);
}

TEST(ClusteringTest, TestKMeans)
{
    DeviceResourceHandle device_resource_handle;
    CreateDeviceResourceHandle(&device_resource_handle);

    DeviceMemoryResource mr;
    UseArenaMemoryResource(&mr, 1024 * 1024);

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
    size_t num_col = 30;

    std::vector<int> labels(num_row);
    std::vector<float> centroids(5 * num_col);
    float inertia = 0.0;
    int n_iter = 0;
    {
        auto res = KmeansFit(
            device_resource_handle,
            feature.data(),
            num_row,
            num_col,
            5,
            10,
            0.0,
            0,
            0,
            42,
            4,
            labels.data(),
            centroids.data(),
            &inertia,
            &n_iter);
    }

    ResetMemoryResource(mr, 2);

    FreeDeviceResourceHandle(device_resource_handle);
}
