#include "engine_impl.h"


ParametricEngine::ParametricEngine()  : impl_(std::make_unique<Impl>()) {}
ParametricEngine::~ParametricEngine() = default;

Bifurcation1DResult ParametricEngine::run_bifurcation_1d(const Bifurcation1DRequest& req) {
    return impl_->run_bif1d(req);
}

Dft1DResult ParametricEngine::run_dft_1d(const Dft1DRequest& req) {
    return impl_->run_dft_1d(req);
}

Bifurcation2DResult ParametricEngine::run_bifurcation_2d(const Bifurcation2DRequest& req) {
    return impl_->run_bif2d(req);
}

LLE1DResult ParametricEngine::run_lle_1d(const LLE1DRequest& req) {
    return impl_->run_lle_1d(req);
}

LS1DResult ParametricEngine::run_ls_1d(const LS1DRequest& req) {
    return impl_->run_ls_1d(req);
}

LLE2DResult ParametricEngine::run_lle_2d(const LLE2DRequest& req) {
    return impl_->run_lle_2d(req);
}

LS2DResult ParametricEngine::run_ls_2d(const LS2DRequest& req) {
    return impl_->run_ls_2d(req);
}

BasinsResult ParametricEngine::run_basins(const BasinsRequest& req) {
    return impl_->run_basins(req);
}

BasinsReclusterResult ParametricEngine::run_basins_recluster(const BasinsReclusterRequest& req) {
    return impl_->run_basins_recluster(req);
}

OrderResult ParametricEngine::run_order(const OrderRequest& req) {
    return impl_->run_order(req);
}

FastSyncResult ParametricEngine::run_fastsync(const FastSyncRequest& req) {
    return impl_->run_fastsync(req);
}
