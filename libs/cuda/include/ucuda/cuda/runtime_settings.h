#pragma once
#include "ucuda/domain/analysis_types.h"
void       set_peak_config(const PeakConfig& c);   // бампает epoch
PeakConfig get_peak_config();
uint64_t   peak_config_epoch();
void set_nvrtc_fmad(bool enabled);
bool get_nvrtc_fmad();
