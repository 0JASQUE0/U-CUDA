#include "expression_internal.h"
using namespace ucuda::equations_detail;
std::string codegen_scheme(const System& s, Scheme sch) {
    if (const int K = gbs_stage_count(sch)) return scheme_gbs(s, K);
    switch (sch) {
    case Scheme::Euler:            return scheme_euler(s);
    case Scheme::EulerCromer:      return scheme_euler_cromer(s);
    case Scheme::ExplicitMidpoint: return scheme_midpoint(s);
    case Scheme::RK4:              return scheme_rk4(s);
    case Scheme::DOPRI78:          return scheme_dopri78(s);
    case Scheme::DOPRI78Legacy:    return scheme_dopri78(s, true);
    case Scheme::RK45:             return scheme_erk(s, &kRK45A[0][0], 6, &kRK45B[0][0], 6);
    case Scheme::DOP853:           return scheme_erk(s, &kDop853A[0][0], 11, kDop853B, 12);
    case Scheme::CD:               return scheme_cd(s);
    case Scheme::ComplexCD:        return scheme_complex_cd(s);
    case Scheme::ComplexCD4:       return scheme_complex_cd4(s);
    case Scheme::ComplexCD4S3:     return scheme_complex_cd4_s3(s);
    case Scheme::ComplexCD4S4:     return scheme_complex_cd4_s4(s);
    case Scheme::ComplexCD4SS01:   return scheme_complex_cd4_ss01(s);
    case Scheme::ComplexCD4SS10:   return scheme_complex_cd4_ss10(s);
    case Scheme::CD10:             return scheme_cd_common(s, CdKind::Real,  false, true);
    case Scheme::ComplexCD10:      return scheme_cd_common(s, CdKind::Cx,    false, true);
    case Scheme::ComplexCD4_10:    return scheme_cd_common(s, CdKind::Cx4,   false, true);
    case Scheme::ComplexCD4S3_10:  return scheme_cd_common(s, CdKind::Cx4s3, false, true);
    case Scheme::ComplexCD4S4_10:  return scheme_cd_common(s, CdKind::Cx4s4, false, true);
    case Scheme::ImplicitEuler:    return scheme_implicit_euler(s);
    case Scheme::ImplicitMidpoint: return scheme_implicit_midpoint(s);
    case Scheme::SEMP:             return scheme_semp(s);
    case Scheme::SIMP:             return scheme_simp(s);
    case Scheme::D:                return scheme_d(s);
    case Scheme::ComplexIEuler:    return scheme_complex_ieuler(s);
    case Scheme::Map:              return scheme_map(s);
    default:                       break;   // ГБШ разобран выше
    }
    throw std::runtime_error("unknown scheme");
}
