
// #include <thrust/host_vector.h>
// #include <thrust/system/cuda/experimental/pinned_allocator.h>

// namespace cuml4c
// {

//     template <typename T>
//     using pinned_host_vector =
//         thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>;

//     namespace traits
//     {
//         template <typename T>
//         class Exporter
//         {
//         public:
//             Exporter(void *x) : t(x) {}
//             inline T get() { return t; }

//         private:
//             T t;
//         };

//         template <typename T>
//         class RangeExporter
//         {
//         public:
//             typedef typename T::value_type r_export_type;

//             RangeExporter(void *x) : object(x) {}
//             ~RangeExporter() {}

//             T get()
//             {
//                 T vec(::Rf_length(object));
//                 ::Rcpp::internal::export_range(object, vec.begin());
//                 return vec;
//             }

//         private:
//             void *object;
//         };

//         template <template <class> class Container, typename T>
//         struct pinned_container_exporter
//         {
//             using type = RangeExporter<Container<T>>;
//         };

//         // enable range exporter for pinned_host_vector
//         template <typename T>
//         class Exporter<cuml4c::pinned_host_vector<T>>
//             : public pinned_container_exporter<cuml4c::pinned_host_vector, T>::type
//         {
//         public:
//             Exporter(void *x)
//                 : pinned_container_exporter<cuml4c::pinned_host_vector, T>::type(x) {}
//         };

//     } // namespace traits

// } // namespace cuml4c
