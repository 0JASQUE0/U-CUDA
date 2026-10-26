#pragma once
// Оценка времени счёта ТРАЕКТОРИЙ — без пост-обработки (пики, DBSCAN, Грам-Шмидт, редукции):
//   T = steps * ops * max(w * ns_thr, ns_lat),
// steps — вызовов шага КРС на нить, ops — цена шага по PTX (krs_step_cost), w — варпов на SM
// при раскладке нитей блоками по варпу, ns_* — калибровка карты (gpu_calibrate). Калибровка
// делается один раз на карту и тип numb и хранится в <exe>\gpu_calibration.txt.
// Адаптивный шаг и CPU-ветки не оцениваются: число шагов там заранее неизвестно, а CPU — другая
// машина счёта.
#include "parametric_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Объём работы одного расчёта. Заполняют сессии (traj_work) из того же запроса, что уходит в движок.
struct TrajWork {
    bool        valid = false;
    std::string why;              // почему оценки нет, если valid == false
    std::string krs_body;
    int         amountOfX = 0;
    double      threads = 0.0;    // параллельных нитей (точек сетки; 1 у continuation на GPU)
    double      steps   = 0.0;    // вызовов шага КРС на нить, с клонами LLE/LS
};

// Среднее 1/h по оси свипа h: число шагов у каждой точки своё, ~ T/h.
inline double traj_mean_inv_h(double lo, double hi, int n, bool log_scale) {
    if (n <= 1) return lo > 0.0 ? 1.0 / lo : 0.0;
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double f = (double)i / (double)(n - 1);
        const double h = (log_scale && lo > 0.0 && hi > 0.0) ? lo * std::pow(hi / lo, f)
                                                               : lo + (hi - lo) * f;
        if (h <= 0.0) return 0.0;
        s += 1.0 / h;
    }
    return s / (double)n;
}

// Шагов на отрезке t при (среднем) 1/h.
inline double traj_steps(double t, double inv_h) {
    return (t > 0.0 && inv_h > 0.0) ? t * inv_h : 0.0;
}

// Подпись времени: 850 ms, 12.3 s, 4m 05s, 2h 10m.
inline std::string traj_format_seconds(double s) {
    char b[64];
    if (!(s >= 0.0) || !std::isfinite(s)) return "?";
    if (s < 1.0)          std::snprintf(b, sizeof b, "%.0f ms", std::max(1.0, s * 1e3));
    else if (s < 60.0)    std::snprintf(b, sizeof b, "%.1f s", s);
    else if (s < 3600.0)  std::snprintf(b, sizeof b, "%dm %02ds", (int)(s / 60.0), (int)std::fmod(s, 60.0));
    else if (s < 86400.0) std::snprintf(b, sizeof b, "%dh %02dm", (int)(s / 3600.0), (int)std::fmod(s / 60.0, 60.0));
    else                  std::snprintf(b, sizeof b, "%.1f days", s / 86400.0);
    return b;
}

class TrajTimeEstimator {
public:
    enum class State { Ready, Pending, Unavailable };
    struct Result {
        State       state = State::Unavailable;
        double      seconds = 0.0;
        bool        rough = false;   // шаг с циклом: посчитан один проход (Ньютон, CD FP)
        std::string detail;          // для подсказки: из чего сложилось число / почему его нет
    };

    TrajTimeEstimator() = default;
    TrajTimeEstimator(const TrajTimeEstimator&) = delete;
    TrajTimeEstimator& operator=(const TrajTimeEstimator&) = delete;
    ~TrajTimeEstimator() {
        if (cost_future_.valid())  cost_future_.wait();
        if (calib_future_.valid()) calib_future_.wait();
    }

    // Не блокирует. Недостающее (цену шага, калибровку) запускает в фоне; калибровку — только при
    // gpu_idle, иначе её замер шёл бы вперемешку с расчётом.
    Result estimate(const TrajWork& w, bool gpu_idle) {
        harvest();
        Result r;
        if (!w.valid) { r.detail = w.why; return r; }
        if (w.krs_body.empty() || w.threads <= 0.0 || w.steps <= 0.0) { r.detail = "nothing to integrate"; return r; }

        if (calib_state_ == 0 && gpu_idle) start_calibration();
        if (calib_state_ == 3) { r.detail = "GPU calibration failed: " + calib_err_; return r; }

        const std::string key = cost_key(w);
        auto it = costs_.find(key);
        if (it == costs_.end()) {
            auto fe = cost_errors_.find(key);
            if (fe != cost_errors_.end()) { r.detail = "step cost: " + fe->second; return r; }
            if (!cost_future_.valid()) start_cost(key, w.krs_body, w.amountOfX);
            r.state = State::Pending;
            r.detail = "compiling the step to count its operations...";
            return r;
        }
        if (calib_state_ != 2) {
            r.state = State::Pending;
            r.detail = calib_state_ == 1 ? "calibrating the GPU..." : "waiting for the GPU to calibrate";
            return r;
        }

        const KrsStepCost& c = it->second;
        r.state   = State::Ready;
        r.seconds = raw_seconds(w, c) * calib_.kernel_factor;
        r.rough   = c.loops;
        char b[640];
        std::snprintf(b, sizeof b,
                      "Trajectories only (no peaks, clustering or post-processing).\n"
                      "%.0f threads x %.4g steps x %.0f ops/step%s\n"
                      "%s: %d SM, %.3g ns/op/warp saturated (+%.0f ops/step), %.3g ns/op single warp,\n"
                      "real kernels x%.2f (refined by finished LLE / LS / Basins runs)",
                      w.threads, w.steps, c.ops,
                      c.loops ? "\nThe step has a loop (Newton / fixed-point iterations): the whole loop body "
                                "is counted once, rarely taken branches included - usually an overestimate." : "",
                      calib_.device.c_str(), calib_.sm_count, calib_.ns_thr, calib_.c0_thr, calib_.ns_lat,
                      calib_.kernel_factor);
        r.detail = b;
        return r;
    }

    // Уточнить kernel_factor по завершённому расчёту, где почти всё время — траектории (LLE, LS,
    // Basins). Не учитываются короткие прогоны (< 2 с: велика доля запуска), шаги с циклом (их
    // счёт — оценка сверху) и выбросы дальше чем в 1.7 раза от текущего значения — так в среднее
    // не попадает прогон, в который вошла компиляция модуля.
    void learn(const TrajWork& w, double real_seconds) {
        harvest();
        if (calib_state_ != 2 || !w.valid || !(real_seconds >= 2.0)) return;
        auto it = costs_.find(cost_key(w));
        if (it == costs_.end() || it->second.loops) return;
        const double raw = raw_seconds(w, it->second);
        if (!(raw > 0.0)) return;
        const double obs = real_seconds / raw;
        const double rel = obs / calib_.kernel_factor;
        if (rel < 1.0 / 1.7 || rel > 1.7) return;
        // Среднее в логарифмах: усредняется отношение времён.
        calib_.kernel_factor = std::exp(0.7 * std::log(calib_.kernel_factor) + 0.3 * std::log(obs));
        std::vector<GpuCalibration> all = read_file();
        all.erase(std::remove_if(all.begin(), all.end(),
                                 [&](const GpuCalibration& c) { return same_card(c, calib_); }),
                  all.end());
        all.push_back(calib_);
        write_file(all);
    }

    // Забыть калибровку этой карты и замерить заново при следующем простое.
    void recalibrate() {
        harvest();
        if (calib_state_ == 1) return;
        forget_ = true;
        calib_state_ = 0;
    }

private:
    // Время по модели, без поправки на настоящие ядра.
    double raw_seconds(const TrajWork& w, const KrsStepCost& c) const {
        const double warps   = std::ceil(w.threads / 32.0);
        const double per_sm  = std::max(1.0, std::ceil(warps / (double)calib_.sm_count));
        const double ns_step = std::max(per_sm * calib_.ns_thr * (c.ops + calib_.c0_thr),
                                        calib_.ns_lat * (c.ops + calib_.c0_lat));
        return w.steps * ns_step * 1e-9;
    }
    static std::string calib_path() { return gpu_calibration_path(); }

    // Строки файла: device \t numb_bytes \t sm \t ns_thr \t c0_thr \t ns_lat \t c0_lat \t calib_ops \t kernel_factor.
    // Строки старого формата (без c0) пропускаются — такая карта откалибруется заново.
    static std::vector<GpuCalibration> read_file() {
        std::vector<GpuCalibration> v;
        std::ifstream f(calib_path());
        std::string line;
        while (std::getline(f, line)) {
            std::vector<std::string> t;
            std::stringstream ss(line);
            std::string x;
            while (std::getline(ss, x, '\t')) t.push_back(x);
            if (t.size() < 8) continue;
            GpuCalibration c;
            c.device     = t[0];
            c.numb_bytes = std::atoi(t[1].c_str());
            c.sm_count   = std::atoi(t[2].c_str());
            c.ns_thr     = std::atof(t[3].c_str());
            c.c0_thr     = std::atof(t[4].c_str());
            c.ns_lat     = std::atof(t[5].c_str());
            c.c0_lat     = std::atof(t[6].c_str());
            c.calib_ops  = std::atof(t[7].c_str());
            if (t.size() > 8 && std::atof(t[8].c_str()) > 0.0) c.kernel_factor = std::atof(t[8].c_str());
            if (c.sm_count > 0 && c.ns_thr > 0.0 && c.ns_lat > 0.0) v.push_back(c);
        }
        return v;
    }

    static void write_file(const std::vector<GpuCalibration>& v) {
        std::ofstream f(calib_path(), std::ios::trunc);
        for (const GpuCalibration& c : v) {
            char b[512];
            std::snprintf(b, sizeof b, "%s\t%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.6g\n", c.device.c_str(),
                          c.numb_bytes, c.sm_count, c.ns_thr, c.c0_thr, c.ns_lat, c.c0_lat, c.calib_ops,
                          c.kernel_factor);
            f << b;
        }
    }

    static bool same_card(const GpuCalibration& a, const GpuCalibration& b) {
        return a.device == b.device && a.numb_bytes == b.numb_bytes && a.sm_count == b.sm_count;
    }

    void start_calibration() {
        calib_state_ = 1;
        const bool forget = forget_;
        forget_ = false;
        calib_future_ = std::async(std::launch::async, [this, forget]() -> bool {
            GpuCalibration me;
            if (!gpu_device_info(me.device, me.sm_count, calib_tmp_err_)) return false;
            me.numb_bytes = (int)sizeof(numb);
            std::vector<GpuCalibration> all = read_file();
            double learned = 0.0;   // повторная калибровка не сбрасывает выученную поправку
            for (const GpuCalibration& c : all)
                if (same_card(c, me)) {
                    if (!forget) { calib_tmp_ = c; return true; }
                    learned = c.kernel_factor;
                }
            if (!gpu_calibrate(calib_tmp_, calib_tmp_err_)) return false;
            if (learned > 0.0) calib_tmp_.kernel_factor = learned;
            all.erase(std::remove_if(all.begin(), all.end(),
                                     [&](const GpuCalibration& c) { return same_card(c, calib_tmp_); }),
                      all.end());
            all.push_back(calib_tmp_);
            write_file(all);
            return true;
        });
    }

    std::string cost_key(const TrajWork& w) const {
        return std::to_string(std::hash<std::string>{}(w.krs_body)) + ":" + std::to_string(w.amountOfX)
             + (get_nvrtc_fmad() ? ":fm1" : ":fm0");
    }

    void start_cost(const std::string& key, const std::string& body, int n) {
        cost_key_tmp_ = key;
        cost_future_ = std::async(std::launch::async, [this, body, n]() -> bool {
            return krs_step_cost(body, n, cost_tmp_, cost_tmp_err_);
        });
    }

    void harvest() {
        if (cost_future_.valid() &&
            cost_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            if (cost_future_.get()) costs_[cost_key_tmp_] = cost_tmp_;
            else                    cost_errors_[cost_key_tmp_] = cost_tmp_err_;
        }
        if (calib_future_.valid() &&
            calib_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            if (calib_future_.get()) { calib_ = calib_tmp_; calib_state_ = 2; }
            else                     { calib_err_ = calib_tmp_err_; calib_state_ = 3; }
        }
    }

    // Пишут только фоновые задачи, читает UI-поток после готовности future.
    std::map<std::string, KrsStepCost> costs_;
    std::map<std::string, std::string> cost_errors_;
    std::future<bool> cost_future_;
    std::string       cost_key_tmp_;
    KrsStepCost       cost_tmp_;
    std::string       cost_tmp_err_;

    int               calib_state_ = 0;   // 0 — нет, 1 — идёт, 2 — есть, 3 — ошибка
    bool              forget_ = false;
    GpuCalibration    calib_;
    std::string       calib_err_;
    std::future<bool> calib_future_;
    GpuCalibration    calib_tmp_;
    std::string       calib_tmp_err_;
};
