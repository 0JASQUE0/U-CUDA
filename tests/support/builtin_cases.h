#pragma once
#include "ucuda/equations/codegen.hpp"
#include <vector>
#include <string>
#include <stdexcept>

namespace regression {
struct BuiltinCase { Scheme scheme; const char* name; };
inline const std::vector<BuiltinCase>& builtin_cases() {
    static const std::vector<BuiltinCase> cases = {
#define CASE(name) {Scheme::name, #name}
        CASE(Euler), CASE(EulerCromer), CASE(ExplicitMidpoint), CASE(RK4), CASE(DOPRI78), CASE(CD),
        CASE(ComplexCD), CASE(ComplexCD4), CASE(ComplexCD4S3), CASE(ComplexCD4S4), CASE(ComplexCD4SS01),
        CASE(ComplexCD4SS10), CASE(CD10), CASE(ComplexCD10), CASE(ComplexCD4_10), CASE(ComplexCD4S3_10),
        CASE(ComplexCD4S4_10), CASE(ImplicitEuler), CASE(ImplicitMidpoint), CASE(SEMP), CASE(SIMP), CASE(D),
        CASE(ComplexIEuler), CASE(GBS), CASE(GBS24), CASE(GBS246), CASE(GBS2468), CASE(GBS246810),
        CASE(GBS24681012), CASE(DOPRI78Legacy), CASE(RK45), CASE(DOP853), CASE(Map), CASE(CDFast), CASE(CDFP), CASE(CDFastFP)
#undef CASE
    };
    return cases;
}
inline std::string builtin_case_name(int value) {
    for (const auto& c : builtin_cases()) if ((int)c.scheme == value) return c.name;
    throw std::runtime_error("Unknown builtin test case");
}
inline std::string builtin_display_name(Scheme method) {
    static const char* names[] = {"Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD",
        "Complex CD", "Complex CD4", "CCD4 (o4s3)", "CCD4 (o4s4)", "CCD4 (s-split)_{0-1}",
        "CCD4 (s-split)_{1-0}", "CD_{1-0}", "Complex CD_{1-0}", "Complex CD4_{1-0}",
        "CCD4 (o4s3)_{1-0}", "CCD4 (o4s4)_{1-0}", "Implicit Euler", "Implicit Midpoint", "SEMP", "SIMP", "D",
        "Complex Implicit Euler", "GBS (n=2)", "GBS 2-4", "GBS 2-4-6", "GBS 2-4-6-8", "GBS 2-4-6-8-10",
        "GBS 2-4-6-8-10-12", "DOPRI78 (legacy)", "RK45", "DOP853", "Map", "CD fast", "CD FP", "CD fast FP"};
    const int index = (int)method;
    if (index < 0 || index >= (int)(sizeof(names) / sizeof(*names))) throw std::runtime_error("Unregistered scheme name");
    return names[index];
}
}
