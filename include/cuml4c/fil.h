
#ifdef __cplusplus
#define EXTERN_C extern "C"
#include <cstddef>
#else
#define EXTERN_C
#include <stdbool.h>
#endif

typedef void *FILModelHandle;

EXTERN_C int FILLoadModel(int const model_type, const char *filename,
                          int const algo, bool const classification,
                          float const threshold, int const storage_type,
                          int const blocks_per_sm, int const threads_per_tree,
                          int const n_items, FILModelHandle *out);

EXTERN_C int FILModelFree(FILModelHandle handle);

EXTERN_C int FILGetNumClasses(FILModelHandle model, size_t *out);

EXTERN_C int FILPredict(FILModelHandle model, const float *x, size_t num_row,
                        bool const output_class_probabilities, float *out);
