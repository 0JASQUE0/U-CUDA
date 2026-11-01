#define main upstream_dae_main
#include "u-cuda/dae_test.cpp"
#undef main
#include <gtest/gtest.h>

TEST(DaeCpu,FixedMethodsAndConstraints) { ASSERT_NO_THROW(cpu_tests()); }
TEST(DaeCpu,AdaptiveAccuracy) { ASSERT_NO_THROW(adaptive_cpu()); }
TEST(DaeCpu,AdaptiveStepDoubling) { ASSERT_NO_THROW(adaptive_doubling_cpu()); }
TEST(DaeCpu,BdfOrdersAndIndices) { ASSERT_NO_THROW(bdf_tests()); }
TEST(DaeCpu,BdfFixedStep) { ASSERT_NO_THROW(bdf_fixed_cpu()); }

class DaeGpu : public ::testing::Test {
    void SetUp() override { ASSERT_EQ(cuInit(0), CUDA_SUCCESS); set_nvrtc_rdc(true); }
    void TearDown() override { set_nvrtc_rdc(false); }
};
TEST_F(DaeGpu,FixedMethodsAndSweeps) { ASSERT_NO_THROW(gpu_tests()); }
TEST_F(DaeGpu,AdaptiveAccuracy) { ASSERT_NO_THROW(adaptive_gpu()); }
TEST_F(DaeGpu,BdfBackendAgreement) { ASSERT_NO_THROW(bdf_gpu()); }
TEST_F(DaeGpu,BdfMinimumStepAndTransient) { ASSERT_NO_THROW(bdf_hmin_and_skip()); }
TEST_F(DaeGpu,Dft) { ASSERT_NO_THROW(dft_dae()); }
TEST_F(DaeGpu,BdfIndexThreeSweep) { ASSERT_NO_THROW(bdf_index3_sweep()); }
TEST_F(DaeGpu,RejectsNoRoot) { ASSERT_NO_THROW(dae_no_root_point()); }
TEST_F(DaeGpu,BdfSweeps) { ASSERT_NO_THROW(dae_sweeps(bdf_sweep_request, "BDF 1-5", true)); }
TEST_F(DaeGpu,RodasSweeps) { ASSERT_NO_THROW(dae_sweeps(rodas_sweep_request, "adaptive DAE Rodas4", false)); }
TEST_F(DaeGpu,StepDoublingAndFixedStep) { ASSERT_NO_THROW(doubling_and_fixed_gpu()); }
TEST_F(DaeGpu,TrapezoidalSweeps) { ASSERT_NO_THROW(dae_sweeps(trap_sweep_request, "adaptive DAE Trapezoidal", false)); }
TEST_F(DaeGpu,BdfFixedSweeps) { ASSERT_NO_THROW(dae_sweeps(bdf_fixed_sweep_request, "BDF fixed", true)); }
