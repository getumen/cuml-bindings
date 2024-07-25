#include "cuml4c/fil.h"
#include "device_resource_handle.cuh"

#include <rmm/device_uvector.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <treelite/c_api.h>
#include <cuml/fil/fil.h>

#include <memory>
#include <string>
#include <fstream>
#include <iterator>

namespace
{

  enum class ModelType
  {
    XGBoost,
    XGBoostJSON,
    LightGBM
  };

  struct FILModel
  {
    __host__ FILModel(std::unique_ptr<ML::fil::forest32_t> forest,
                      int const num_features)
        : forest_(std::move(forest)),
          numFeatures_(num_features) {}

    std::unique_ptr<ML::fil::forest32_t> forest_;
    int const numFeatures_;
  };

  __host__ int treeliteLoadModel(ModelType const model_type,
                                 char const *filename,
                                 TreeliteModelHandle *model_handle)
  {
    std::string json_config = "{\"allow_unknown_field\": True}";
    switch (model_type)
    {
    case ModelType::XGBoost:
      return TreeliteLoadXGBoostModel(filename, json_config.c_str(), model_handle);
    case ModelType::XGBoostJSON: {
      std::ifstream file(filename); // Replace with your file name
      if (!file.is_open()) {
          return -1;
      }
      std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
      file.close(); 

      return TreeliteLoadXGBoostModelFromString(content.c_str(), content.length(), json_config.c_str(), model_handle);
    }
    case ModelType::LightGBM:
      return TreeliteLoadLightGBMModel(filename, json_config.c_str(), model_handle);
    }

    // unreachable
    return -1;
  }

} // namespace

__host__ int FILLoadModel(
    const DeviceResourceHandle handle,
    int model_type,
    const char *filename,
    int algo,
    bool classification,
    float threshold,
    int storage_type,
    int blocks_per_sm,
    int threads_per_tree,
    int n_items,
    FILModelHandle *out)
{
  auto handle_p = static_cast<cuml4c::DeviceResource *>(handle);

  TreeliteModelHandle model_handle;
  {
    auto const res = treeliteLoadModel(
        /*model_type=*/static_cast<ModelType>(model_type),
        /*filename=*/filename,
        &model_handle);
    if (res < 0)
    {
      return FIL_FAIL_TO_LOAD_MODEL;
    }
  }

  int num_features = 0;
  {
    auto res = TreeliteQueryNumFeature(model_handle, &num_features);
    if (res < 0)
    {
      return FIL_FAIL_TO_GET_NUM_FEATURE;
    }
  }

  ML::fil::treelite_params_t params;
  params.algo = static_cast<ML::fil::algo_t>(algo);
  params.output_class = classification;
  params.threshold = threshold;
  params.storage_type = static_cast<ML::fil::storage_type_t>(storage_type);
  params.blocks_per_sm = blocks_per_sm;
  params.output_class = classification;
  params.threads_per_tree = threads_per_tree;
  params.n_items = n_items;
  params.pforest_shape_str = nullptr;
  params.precision = ML::fil::precision_t::PRECISION_FLOAT32;

  ML::fil::forest_variant f;

  ML::fil::from_treelite(
      /*handle=*/*handle_p->handle,
      /*pforest=*/&f,
      /*model=*/model_handle,
      /*tl_params=*/&params);

  auto forest = std::make_unique<ML::fil::forest32_t>(std::move(std::get<ML::fil::forest32_t>(f)));

  auto model = std::make_unique<FILModel>(
      std::move(forest),
      num_features);

  *out = static_cast<FILModelHandle>(model.release());

  {
    auto res = TreeliteFreeModel(model_handle);
    if (res < 0)
    {
      return FIL_FAIL_TO_FREE_MODEL;
    }
  }

  return FIL_SUCCESS;
}

__host__ int FILFreeModel(
    const DeviceResourceHandle handle,
    FILModelHandle model)
{
  auto handle_p = static_cast<cuml4c::DeviceResource *>(handle);
  auto model_ptr = static_cast<FILModel const *>(model);
  ML::fil::free(*handle_p->handle, *model_ptr->forest_);
  delete model_ptr;
  return FIL_SUCCESS;
}

__host__ int FILPredict(
    const DeviceResourceHandle handle,
    FILModelHandle model,
    const float *x,
    size_t num_row,
    bool output_class_probabilities,
    float *preds)
{
  auto handle_p = static_cast<cuml4c::DeviceResource *>(handle);

  auto fil_model = static_cast<FILModel *>(model);

  auto d_x = rmm::device_uvector<float>(
      fil_model->numFeatures_ * num_row,
      handle_p->handle->get_stream());

  raft::update_device(d_x.data(),
                      x,
                      fil_model->numFeatures_ * num_row,
                      handle_p->handle->get_stream());

  auto pred_size = output_class_probabilities
                       ? 2 * num_row
                       : num_row;

  auto d_preds = rmm::device_uvector<float>(
      pred_size,
      handle_p->handle->get_stream());

  ML::fil::predict(/*h=*/*handle_p->handle,
                   /*f=*/*fil_model->forest_,
                   /*preds=*/d_preds.begin(),
                   /*data=*/d_x.begin(),
                   /*num_rows=*/num_row,
                   /*predict_proba=*/output_class_probabilities);

  raft::update_host(preds,
                    d_preds.begin(),
                    d_preds.size(),
                    handle_p->handle->get_stream());

  handle_p->handle->sync_stream();

  return FIL_SUCCESS;
}
