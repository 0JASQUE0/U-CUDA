#pragma once
#include "ucuda/domain/adaptive_solver.h"
#include "ucuda/equations/codegen.hpp"

// The generated body's prefix is also used by the persistent phase solver.
std::string codegen_adaptive(const System& sys, const std::string& scheme,
    const std::string& base_body, bool custom, const adaptive::Options& options);
bool is_adaptive_body(const std::string& body);
std::string adaptive_model_prefix(const std::string& body);
std::string adaptive_phase_loop(const std::string& body);
std::string adaptive_ensemble_function(const std::string& body, bool cpu);

std::string adaptive_sync_function(const std::string& body);
