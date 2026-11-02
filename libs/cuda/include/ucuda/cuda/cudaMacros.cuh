#pragma once
#include "ucuda/domain/configCUDA.h"

// Под NVRTC (__CUDACC_RTC__ определён автоматически) host-only заголовки
// недоступны и не нужны — kernel-ы их не используют.
#ifndef __CUDACC_RTC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#endif
