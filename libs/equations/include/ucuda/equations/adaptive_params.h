#pragma once
#include "ucuda/domain/adaptive_settings.h"
#include "ucuda/equations/codegen.hpp"
inline bool adaptive_scheme_name_ok(const std::string& scheme) {
    return adaptive_scheme_name_supported(scheme);
}

inline int adaptive_scheme_q(const std::string& scheme) {
    return adaptive_scheme_q_guess(scheme);
}

inline bool adaptive_build_params(const AdaptiveSettings& s, const AdaptiveCode& code, int n,
                                  double span, UcudaAdaptParams& P, std::string& err,
                                  std::string* ctrl_body = nullptr) {
    if (ctrl_body) ctrl_body->clear();
    std::memset(&P, 0, sizeof(P));
    if (n < 1 || n > UCUDA_AD_MAXN) { err = "adaptive step: dimension out of range"; return false; }
    double rtol = 0;
    if (!parse_num_checked(s.rtol, rtol) || !(rtol >= 0)) { err = "rtol must be a number >= 0"; return false; }
    std::vector<double> atol;
    if (!adaptive_parse_atol(s.atol, n, atol, err)) return false;
    bool any_tol = rtol > 0;
    for (double v : atol) any_tol |= v > 0;
    if (!any_tol) { err = "rtol and atol cannot both be zero"; return false; }
    double h0 = 0, hmin = 0, hmax = 0;
    if (!adaptive_parse_opt(s.h0, "h0", h0, err)) return false;
    if (!adaptive_parse_opt(s.hmin, "h_min", hmin, err)) return false;
    if (!adaptive_parse_opt(s.hmax, "h_max", hmax, err)) return false;
    if (hmin > 0 && hmax > 0 && hmin > hmax) { err = "h_min > h_max"; return false; }
    double maxrej = 0;
    bool maxrej_blank = true;
    for (char ch : s.max_rej) if (!std::isspace((unsigned char)ch)) { maxrej_blank = false; break; }
    if (!maxrej_blank && (!parse_num_checked(s.max_rej, maxrej) || !(maxrej >= 0) || maxrej > 1e9
                          || (double)(long long)maxrej != maxrej)) {
        err = "max rejects must be empty or a whole number >= 0";
        return false;
    }

    AdaptiveCtrlResolved cr;
    if (!adaptive_resolve_ctrl(s.ctrl, code.q, cr, err)) return false;
    std::vector<double> c = cr.def;
    // Параметры другого регулятора (имя заменено adaptive_ctrl_effective) к Хайреру не подходят.
    if (!s.ctrl_params.empty() && adaptive_ctrl_effective(s.ctrl) == s.ctrl) {
        if (s.ctrl_params.size() != cr.par.size()) { err = "controller parameters: wrong count"; return false; }
        for (size_t i = 0; i < cr.par.size(); ++i)
            if (!parse_num_checked(s.ctrl_params[i], c[i])) {
                err = "controller parameter " + cr.par[i] + " is not a number";
                return false;
            }
    }
    if (cr.id == UCUDA_CTRL_CUSTOM) {
        std::string prep, body;
        adaptive_ctrl_unpack(cr.body, prep, body);
        bool blank = true;
        for (char ch : body) if (!std::isspace((unsigned char)ch)) { blank = false; break; }
        if (blank) { err = "controller '" + s.ctrl + "' has an empty body"; return false; }
        if (ctrl_body) *ctrl_body = cr.body;
    }

    P.rtol = rtol;
    for (int i = 0; i < n; ++i) P.atol[i] = atol[i];
    P.h0 = h0; P.hmin = hmin; P.span = span;
    // Без ограничения — +inf, а не 0: попытка шага зажимает h одним сравнением h > hmax.
    P.hmax = hmax > 0 ? (numb)hmax : std::numeric_limits<numb>::infinity();
    P.maxrej = maxrej_blank ? 0 : (maxrej == 0 ? UCUDA_AD_NO_RETRY : (int)maxrej);
    for (size_t i = 0; i < c.size() && i < UCUDA_CTL_NPAR; ++i) P.c[i] = c[i];
    P.q = code.q; P.nlo = code.nlo; P.ctrl = cr.id;
    P.clip   = cr.hairer_rules ? 1 : 0;
    P.h0mode = cr.hairer_rules ? 1 : 0;
    P.emb_rhs = code.emb_rhs; P.dprep_rhs = code.dprep_rhs;
    return true;
}
