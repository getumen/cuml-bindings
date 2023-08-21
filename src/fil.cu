#include "async_utils.cuh"
#include "cuda_utils.h"
#include "fil_utils.h"
#include "handle_utils.h"
#include "preprocessor.h"
#include "stream_allocator.h"
#include "treelite_utils.cuh"
#include "cuml4c/fil.h"

#include <cuml/fil/fil.h>
#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <treelite/c_api.h>

#include <memory>
#include <string>

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
    __host__ FILModel(std::unique_ptr<raft::handle_t> handle,
                      cuml4c::fil::forest_uptr forest,
                      size_t const num_classes,
                      size_t const num_features)
        : handle_(std::move(handle)), forest_(std::move(forest)),
          numClasses_(num_classes), numFeatures_(num_features) {}

    std::unique_ptr<raft::handle_t> const handle_;
    // NOTE: the destruction of `forest_` must precede the destruction of
    // `handle_`.
    cuml4c::fil::forest_uptr forest_;
    size_t const numClasses_;
    size_t const numFeatures_;
  };

  __host__ int treeliteLoadModel(ModelType const model_type, char const *filename,
                                 cuml4c::TreeliteHandle &tl_handle)
  {
    switch (model_type)
    {
    case ModelType::XGBoost:
      return TreeliteLoadXGBoostModel(filename, tl_handle.get());
    case ModelType::XGBoostJSON:
      return TreeliteLoadXGBoostJSON(filename, tl_handle.get());
    case ModelType::LightGBM:
      return TreeliteLoadLightGBMModel(filename, tl_handle.get());
    }

    // unreachable
    return -1;
  }

  /*
   * The 'ML::fil::treelite_params_t::threads_per_tree' and
   * 'ML::fil::treelite_params_t::n_items' parameters are only supported in
   * RAPIDS cuML 21.08 or above.
   */
  CUML4C_ASSIGN_IF_PRESENT(threads_per_tree)
  CUML4C_NOOP_IF_ABSENT(threads_per_tree)

  CUML4C_ASSIGN_IF_PRESENT(n_items)
  CUML4C_NOOP_IF_ABSENT(n_items)

} // namespace

__host__ int FILLoadModel(
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

  cuml4c::TreeliteHandle tl_handle;
  {
    auto const rc = treeliteLoadModel(
        /*model_type=*/static_cast<ModelType>(model_type),
        /*filename=*/filename,
        tl_handle);
    if (rc < 0)
    {
      return -1;
    }
  }

  ML::fil::treelite_params_t params;
  params.algo = static_cast<ML::fil::algo_t>(algo);
  params.output_class = classification;
  params.threshold = threshold;
  params.storage_type = static_cast<ML::fil::storage_type_t>(storage_type);
  params.blocks_per_sm = blocks_per_sm;
  params.output_class = classification;
  set_threads_per_tree(params, threads_per_tree);
  set_n_items(params, n_items);
  params.pforest_shape_str = nullptr;

  auto stream_view = cuml4c::stream_allocator::getOrCreateStream();
  auto handle = std::make_unique<raft::handle_t>();
  cuml4c::handle_utils::initializeHandle(*handle, stream_view.value());

  auto forest = cuml4c::fil::make_forest(*handle, /*src=*/[&]
                                         {
    ML::fil::forest *f;
    ML::fil::from_treelite(/*handle=*/*handle, /*pforest=*/&f,
                           /*model=*/*tl_handle.get(),
                           /*tl_params=*/&params);
    return f; });

  size_t num_classes = 0;
  if (classification)
  {
    auto const rc = TreeliteQueryNumClass(/*handle=*/*tl_handle.get(),
                                          /*out=*/&num_classes);
    if (rc < 0)
    {
      return -1;
    }

    // Treelite returns 1 as number of classes for binary classification.
    num_classes = std::max(num_classes, size_t(2));
  }

  size_t num_features = 0;
  {
    auto const rc = TreeliteQueryNumFeature(/*handle=*/*tl_handle.get(),
                                            /*out=*/&num_features);
    if (rc < 0)
    {
      return -1;
    }
  }

  auto model = std::make_unique<FILModel>(
      /*handle=*/std::move(handle),
      std::move(forest),
      num_classes,
      num_features);

  *out = static_cast<FILModelHandle>(model.release());

  return 0;
}

__host__ int FILModelFree(
    FILModelHandle handle)
{
  delete static_cast<FILModel *>(handle);
  return 0;
}

__host__ int FILGetNumClasses(
    FILModelHandle model,
    size_t *out)
{
  auto const model_xptr = static_cast<FILModel const *>(model);
  *out = model_xptr->numClasses_;
  return 0;
}

__host__ int FILPredict(
    FILModelHandle model,
    const float *x,
    size_t num_row,
    bool output_class_probabilities,
    float *out)
{

  auto const fil_model = static_cast<FILModel const *>(model);

  if (output_class_probabilities && fil_model->numClasses_ == 0)
  {
    return -1;
  }

  auto &handle = *(fil_model->handle_);

  auto output_size = output_class_probabilities
                         ? fil_model->numClasses_ * num_row
                         : num_row;

  const auto feature_size = fil_model->numFeatures_ * num_row;
  // ensemble input data
  thrust::device_vector<float> d_x(feature_size);

  // TODO: async copy
  thrust::copy(
      x,
      x + feature_size,
      d_x.begin());

  // ensemble output
  thrust::device_vector<float>
      d_preds(output_size);

  ML::fil::predict(/*h=*/handle,
                   /*f=*/fil_model->forest_.get(),
                   /*preds=*/d_preds.data().get(),
                   /*data=*/d_x.data().get(),
                   /*num_rows=*/num_row,
                   /*predict_proba=*/output_class_probabilities);

  // TODO: async copy
  thrust::copy(
      d_preds.begin(),
      d_preds.end(),
      out);

  return 0;
}
