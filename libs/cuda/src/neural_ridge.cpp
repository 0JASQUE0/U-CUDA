#include "ucuda/cuda/neural_ridge.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <memory>
#include <string>
#include <type_traits>

namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check(int status, const char* operation) {
    if (status != 0) throw std::runtime_error(std::string(operation) + ": status " + std::to_string(status));
}

void check_cancel(const std::atomic<bool>* cancel) {
    if (cancel && cancel->load(std::memory_order_relaxed)) throw std::runtime_error("ridge training was stopped");
}

template<class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count > (std::numeric_limits<size_t>::max)() / sizeof(T)) throw std::runtime_error("ridge buffer is too large");
        check(cudaMalloc(reinterpret_cast<void**>(&ptr_), (std::max)(count, size_t{1}) * sizeof(T)), "ridge cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }

    void upload(const T* source, size_t count, size_t offset = 0) const {
        check(cudaMemcpy(ptr_ + offset, source, count * sizeof(T), cudaMemcpyHostToDevice), "ridge upload");
    }
    void download(T* destination, size_t count) const {
        check(cudaMemcpy(destination, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost), "ridge download");
    }
    void copy_from(const DeviceBuffer& source, size_t count) const {
        check(cudaMemcpy(ptr_, source.ptr_, count * sizeof(T), cudaMemcpyDeviceToDevice), "ridge device copy");
    }

private:
    T* ptr_ = nullptr;
};
}

ReadoutModel neural_ridge_gpu(const std::vector<double>& features, const std::vector<double>& targets,
                            int columns, int classes, const ReadoutSplit& split,
                            const std::vector<double>& lambdas, const std::atomic<bool>* cancel) {
    if (columns < 1 || classes < 1 || lambdas.empty() ||
        split.train0 >= split.train1 || split.train1 > split.val0 || split.val0 >= split.val1 ||
        split.val1 > features.size() / columns || split.val1 > targets.size() / classes ||
        split.train1 - split.train0 > (size_t)(std::numeric_limits<int>::max)())
        throw std::runtime_error("invalid GPU ridge training dimensions or split");
    check_cancel(cancel);

    const int n = (int)(split.train1 - split.train0);
    const bool dual = n < columns;
    const int order = (std::min)(n, columns);
    if (order == (std::numeric_limits<int>::max)()) throw std::runtime_error("ridge matrix is too large");
    const size_t gram_size = (size_t)order * order;
    const size_t rhs_size = (size_t)order * classes;
    const size_t weight_count = (size_t)columns * classes;

    ReadoutModel model;
    model.M = columns;
    model.K = classes;
    readout_detail::feature_normalization(features, split.train0, n, model);
    const auto ymean = readout_detail::target_means(targets, split.train0, n, classes);
    check_cancel(cancel);

    // The runtime uses the CUDA context current on this worker, as does Neural simulation.
    cublasHandle_t blas_handle = nullptr;
    check(cublasCreate(&blas_handle), "cublasCreate for ridge");
    const std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, decltype(&cublasDestroy)> blas(blas_handle, cublasDestroy);
    cusolverDnHandle_t solver_handle = nullptr;
    check(cusolverDnCreate(&solver_handle), "cusolverDnCreate for ridge");
    const std::unique_ptr<std::remove_pointer_t<cusolverDnHandle_t>, decltype(&cusolverDnDestroy)> solver(solver_handle, cusolverDnDestroy);

    // Upload row-major samples in blocks. cuBLAS sees their transpose (columns x n).
    DeviceBuffer<double> x((size_t)n * columns);
    const int block_rows = (std::min)(n, 1024);
    std::vector<double> block((size_t)block_rows * columns);
    for (size_t first = 0; first < (size_t)n; first += block_rows) {
        check_cancel(cancel);
        const size_t rows = (std::min)((size_t)block_rows, (size_t)n - first);
        for (size_t t = 0; t < rows; ++t) {
            const double* row = features.data() + (split.train0 + first + t) * columns;
            for (int i = 0; i < columns; ++i)
                block[t * columns + i] = std::isinf(model.scale[i]) ? 0.0 : (row[i] - model.mean[i]) / model.scale[i];
        }
        x.upload(block.data(), rows * columns, first * columns);
    }
    std::vector<double> centered((size_t)n * classes);
    for (int k = 0; k < classes; ++k)
        for (int t = 0; t < n; ++t)
            centered[(size_t)k * n + t] = targets[(split.train0 + t) * classes + k] - ymean[k];
    DeviceBuffer<double> y(centered.size()), gram(gram_size), factor(gram_size), rhs(rhs_size), answer(rhs_size);
    DeviceBuffer<double> dual_weights(dual ? weight_count : 0), shift(order);
    DeviceBuffer<int> info(1);
    y.upload(centered.data(), centered.size());
    constexpr auto lower = CUBLAS_FILL_MODE_LOWER;
    const double average = 1.0 / n, zero = 0.0, one = 1.0;
    // Dual system: (XX^T/n + lambda I)a = Yc, then W = X^T a/n.
    check(cublasDsyrk(blas.get(), lower, dual ? CUBLAS_OP_T : CUBLAS_OP_N, order, dual ? columns : n,
                      &average, x.get(), columns, &zero, gram.get(), order), "cublasDsyrk for ridge");
    if (!dual)
        check(cublasDgemm(blas.get(), CUBLAS_OP_N, CUBLAS_OP_N, columns, classes, n, &average,
                          x.get(), columns, y.get(), n, &zero, rhs.get(), columns), "cublasDgemm for ridge RHS");

    int workspace_count = 0;
    check(cusolverDnDpotrf_bufferSize(solver.get(), lower, order, factor.get(), order, &workspace_count), "ridge potrf workspace");
    if (workspace_count < 0) throw std::runtime_error("invalid GPU ridge workspace size");
    DeviceBuffer<double> workspace((size_t)workspace_count);
    std::vector<double> weights(weight_count), diagonal(order);
    ReadoutModel candidate = model;
    candidate.w.resize((size_t)classes * ((size_t)columns + 1));
    model.lambda.resize(classes);
    model.val_nrmse.resize(classes);
    for (double lambda : lambdas) {
        check_cancel(cancel);
        factor.copy_from(gram, gram_size);
        answer.copy_from(dual ? y : rhs, rhs_size);
        std::fill(diagonal.begin(), diagonal.end(), lambda);
        shift.upload(diagonal.data(), diagonal.size());
        check(cublasDaxpy(blas.get(), order, &one, shift.get(), 1, factor.get(), order + 1), "ridge diagonal regularization");
        check(cusolverDnDpotrf(solver.get(), lower, order, factor.get(), order, workspace.get(), workspace_count, info.get()),
              "cusolverDnDpotrf for ridge");
        int status = 0;
        info.download(&status, 1);
        check(status, "ridge Cholesky factorization info");
        check_cancel(cancel);
        check(cusolverDnDpotrs(solver.get(), lower, order, classes, factor.get(), order, answer.get(), order, info.get()),
              "cusolverDnDpotrs for ridge");
        info.download(&status, 1);
        check(status, "ridge Cholesky solve info");
        if (dual)
            check(cublasDgemm(blas.get(), CUBLAS_OP_N, CUBLAS_OP_N, columns, classes, n, &average,
                              x.get(), columns, answer.get(), n, &zero, dual_weights.get(), columns), "cublasDgemm for ridge weights");
        (dual ? dual_weights : answer).download(weights.data(), weights.size());
        for (int k = 0; k < classes; ++k) {
            double* row = candidate.w.data() + (size_t)k * ((size_t)columns + 1);
            std::copy_n(weights.data() + (size_t)k * columns, columns, row);
            row[columns] = ymean[k];
        }
        const auto predictions = readout_predict(candidate, features, split.val0, split.val1);
        readout_detail::consider_classifier(model, candidate, targets, split, lambda, predictions);
    }
    check_cancel(cancel);
    return model;
}
