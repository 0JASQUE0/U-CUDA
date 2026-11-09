#pragma once
// Обучаемый выход сети: признаки из записи, разбиение по времени, ridge на CPU, метрики. Сеть и задача разделены:
// здесь ничего не знает ни о спайках, ни о GPU, кроме фильтра журнала. Признаки F[t * M + i], цели Y[t * K + k].

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

// Полуинтервалы отсчётов. Между частями пропуск: окна одной записи не должны оказаться по разные стороны разделения.
struct ReadoutSplit { size_t train0 = 0, train1 = 0, val0 = 0, val1 = 0, test0 = 0, test1 = 0; };

inline ReadoutSplit readout_split(size_t total, size_t washout, double f_train, double f_val, size_t gap) {
    if (washout + 2 * gap >= total) throw std::runtime_error("the record is too short for the split");
    const size_t n = total - washout - 2 * gap;
    const size_t a = (size_t)((double)n * f_train), b = (size_t)((double)n * f_val);
    if (a < 2 || b < 1 || a + b >= n) throw std::runtime_error("every part of the split needs samples");
    ReadoutSplit s;
    s.train0 = washout; s.train1 = washout + a;
    s.val0 = s.train1 + gap; s.val1 = s.val0 + b;
    s.test0 = s.val1 + gap; s.test1 = total;
    return s;
}

struct ReadoutModel {
    int M = 0, K = 0;
    std::vector<double> mean, scale;   // [M], по обучающей части
    std::vector<double> w;             // [K * (M + 1)], последний столбец свободный член
    std::vector<double> lambda;        // [K]
    std::vector<double> val_nrmse;     // [K]
    double val_accuracy = 0.0;         // классификатор: доля верных argmax на валидации
};

// Номер наибольшего выхода в каждой строке [n][K]: класс ridge-классификатора по one-hot.
inline std::vector<int> readout_argmax(const std::vector<double>& p, int K) {
    std::vector<int> out(p.size() / (size_t)K);
    for (size_t t = 0; t < out.size(); ++t) {
        const double* row = p.data() + t * (size_t)K;
        out[t] = (int)(std::max_element(row, row + K) - row);
    }
    return out;
}

// Экспоненциальный фильтр журнала спайков на сетке записи: спайк шага n стоит на времени (n + 1) h, поэтому в точку
// шага s входят только спайки с n + 1 <= s. Журнал упорядочен по шагу. Узлы вне [node0, node0 + M) пропускаются.
inline std::vector<double> readout_spike_features(const std::vector<long long>& step, const std::vector<int>& node, int node0, int M,
                                                  long long first_step, long long stride, int points, double tau_steps) {
    if (M < 1 || points < 1 || stride < 1 || !(tau_steps > 0.0)) throw std::runtime_error("bad spike filter settings");
    std::vector<double> F((size_t)points * M, 0.0), trace((size_t)M, 0.0);
    const double decay = std::exp(-(double)stride / tau_steps);
    size_t e = 0;
    while (e < step.size() && step[e] + 1 <= first_step - stride) ++e;
    for (int p = 0; p < points; ++p) {
        const long long at = first_step + (long long)p * stride;
        for (double& v : trace) v *= decay;
        for (; e < step.size() && step[e] + 1 <= at; ++e) {
            const int i = node[e] - node0;
            if (i >= 0 && i < M) trace[(size_t)i] += std::exp(-(double)(at - step[e] - 1) / tau_steps);
        }
        std::copy(trace.begin(), trace.end(), F.begin() + (size_t)p * M);
    }
    return F;
}

inline double readout_nrmse(const double* y, const double* yhat, size_t stride, size_t n) {
    double m = 0.0;
    for (size_t t = 0; t < n; ++t) m += y[t * stride];
    m /= (double)n;
    double err = 0.0, var = 0.0;
    for (size_t t = 0; t < n; ++t) {
        const double a = y[t * stride], d = a - yhat[t * stride];
        err += d * d;
        var += (a - m) * (a - m);
    }
    return var > 0.0 ? std::sqrt(err / var) : std::numeric_limits<double>::quiet_NaN();
}

// Квадрат коэффициента корреляции цели и выхода: вклад одной задержки в memory capacity.
inline double readout_r2(const double* y, const double* yhat, size_t stride, size_t n) {
    double my = 0.0, mh = 0.0;
    for (size_t t = 0; t < n; ++t) { my += y[t * stride]; mh += yhat[t * stride]; }
    my /= (double)n; mh /= (double)n;
    double cov = 0.0, vy = 0.0, vh = 0.0;
    for (size_t t = 0; t < n; ++t) {
        const double a = y[t * stride] - my, b = yhat[t * stride] - mh;
        cov += a * b; vy += a * a; vh += b * b;
    }
    return vy > 0.0 && vh > 0.0 ? cov * cov / (vy * vh) : 0.0;
}

// Доля верных бит: цель 0 или 1, ответ по порогу 0.5.
inline double readout_bit_accuracy(const double* y, const double* yhat, size_t stride, size_t n) {
    size_t right = 0;
    for (size_t t = 0; t < n; ++t) right += (y[t * stride] > 0.5) == (yhat[t * stride] > 0.5);
    return n ? (double)right / (double)n : std::numeric_limits<double>::quiet_NaN();
}

inline std::vector<double> readout_predict(const ReadoutModel& r, const std::vector<double>& F, size_t t0, size_t t1) {
    const size_t M = (size_t)r.M, K = (size_t)r.K, n = t1 - t0;
    std::vector<double> out(n * K);
    for (size_t t = 0; t < n; ++t) {
        const double* row = F.data() + (t0 + t) * M;
        for (size_t k = 0; k < K; ++k) {
            const double* w = r.w.data() + k * (M + 1);
            double sum = w[M];
            for (size_t i = 0; i < M; ++i) sum += w[i] * (row[i] - r.mean[i]) / r.scale[i];
            out[t * K + k] = sum;
        }
    }
    return out;
}

namespace readout_detail {
template <class Fn> void parallel_for(size_t n, Fn fn) {
    const unsigned threads = (std::max)(1u, std::thread::hardware_concurrency());
    std::atomic<size_t> next{ 0 };
    const auto work = [&]() { for (size_t i; (i = next.fetch_add(1)) < n;) fn(i); };
    std::vector<std::thread> pool;
    for (unsigned q = 1; q < threads; ++q) pool.emplace_back(work);
    work();
    for (std::thread& t : pool) t.join();
}
// Холецкий на месте: сначала столбец левее, поэтому внутренний цикл идёт по строке подряд.
inline void cholesky(std::vector<double>& A, size_t n) {
    for (size_t j = 0; j < n; ++j) {
        double* Aj = A.data() + j * n;
        double d = Aj[j];
        for (size_t k = 0; k < j; ++k) d -= Aj[k] * Aj[k];
        if (!(d > 0.0)) throw std::runtime_error("the ridge matrix is not positive definite");
        d = std::sqrt(d);
        Aj[j] = d;
        parallel_for(n - j - 1, [&](size_t q) {
            double* Ai = A.data() + (j + 1 + q) * n;
            double s = Ai[j];
            for (size_t k = 0; k < j; ++k) s -= Ai[k] * Aj[k];
            Ai[j] = s / d;
        });
    }
}
inline void cholesky_solve(const std::vector<double>& L, size_t n, double* b) {
    for (size_t i = 0; i < n; ++i) {
        const double* Li = L.data() + i * n;
        double s = b[i];
        for (size_t k = 0; k < i; ++k) s -= Li[k] * b[k];
        b[i] = s / Li[i];
    }
    for (size_t i = n; i-- > 0;) {
        double s = b[i];
        for (size_t k = i + 1; k < n; ++k) s -= L[k * n + i] * b[k];
        b[i] = s / L[i * n + i];
    }
}
}  // namespace readout_detail

// Вес на цель берётся при лучшем по валидации lambda. Обучение видит только train: и среднее, и масштаб, и матрицу.
// Система (X^T X / n + lambda I) w = X^T y / n, X стандартизован, y центрирован: lambda не зависит от длины записи.
// classify: Y это one-hot, lambda одна на все выходы и выбирается по accuracy на валидации, при равенстве большая.
inline ReadoutModel readout_fit(const std::vector<double>& F, const std::vector<double>& Y, int M, int K, const ReadoutSplit& s,
                                const std::vector<double>& lambdas, bool classify = false) {
    if (M < 1 || K < 1 || lambdas.empty()) throw std::runtime_error("the readout needs features, targets and lambdas");
    const size_t Ms = (size_t)M, Ks = (size_t)K, n = s.train1 - s.train0;
    ReadoutModel r;
    r.M = M; r.K = K;
    r.mean.assign(Ms, 0.0); r.scale.assign(Ms, 1.0);
    std::vector<double> X(n * Ms);   // по столбцам: скалярные произведения идут подряд
    readout_detail::parallel_for(Ms, [&](size_t i) {
        double m = 0.0, v = 0.0;
        for (size_t t = 0; t < n; ++t) m += F[(s.train0 + t) * Ms + i];
        m /= (double)n;
        for (size_t t = 0; t < n; ++t) { const double d = F[(s.train0 + t) * Ms + i] - m; v += d * d; }
        r.mean[i] = m;
        r.scale[i] = std::sqrt(v / (double)n);
    });
    // Редкий на обучении признак не должен раздуваться стандартизацией: хвост спада старого спайка дал бы масштаб 1e-22, и новый спайк
    // на тесте взорвал бы выход. Поэтому масштаб не ниже сотой доли типичного, а совсем молчащий признак выпадает.
    std::vector<double> sorted(r.scale);
    std::nth_element(sorted.begin(), sorted.begin() + (std::ptrdiff_t)(Ms / 2), sorted.end());
    const double floor_scale = 0.01 * sorted[Ms / 2];
    const double top = *std::max_element(r.scale.begin(), r.scale.end());
    readout_detail::parallel_for(Ms, [&](size_t i) {
        double* col = X.data() + i * n;
        const bool silent = !(r.scale[i] > 1e-9 * top);
        r.scale[i] = silent ? std::numeric_limits<double>::infinity() : (std::max)(r.scale[i], floor_scale);
        for (size_t t = 0; t < n; ++t) col[t] = silent ? 0.0 : (F[(s.train0 + t) * Ms + i] - r.mean[i]) / r.scale[i];
    });
    std::vector<double> G(Ms * Ms), b(Ms * Ks), ymean(Ks, 0.0);
    readout_detail::parallel_for(Ms, [&](size_t i) {
        const double* a = X.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double* c = X.data() + j * n;
            double sum = 0.0;
            for (size_t t = 0; t < n; ++t) sum += a[t] * c[t];
            G[i * Ms + j] = sum / (double)n;
        }
    });
    for (size_t k = 0; k < Ks; ++k) {
        for (size_t t = 0; t < n; ++t) ymean[k] += Y[(s.train0 + t) * Ks + k];
        ymean[k] /= (double)n;
    }
    readout_detail::parallel_for(Ms, [&](size_t i) {
        for (size_t k = 0; k < Ks; ++k) {
            double sum = 0.0;
            const double* a = X.data() + i * n;
            for (size_t t = 0; t < n; ++t) sum += a[t] * (Y[(s.train0 + t) * Ks + k] - ymean[k]);
            b[i * Ks + k] = sum / (double)n;
        }
    });
    r.w.assign(Ks * (Ms + 1), 0.0);
    r.lambda.assign(Ks, 0.0);
    r.val_nrmse.assign(Ks, std::numeric_limits<double>::infinity());
    for (double lam : lambdas) {
        std::vector<double> L(G);
        for (size_t i = 0; i < Ms; ++i) L[i * Ms + i] += lam;
        readout_detail::cholesky(L, Ms);
        ReadoutModel c = r;
        for (size_t k = 0; k < Ks; ++k) {
            std::vector<double> x(Ms);
            for (size_t i = 0; i < Ms; ++i) x[i] = b[i * Ks + k];
            readout_detail::cholesky_solve(L, Ms, x.data());
            std::copy(x.begin(), x.end(), c.w.begin() + (std::ptrdiff_t)(k * (Ms + 1)));
            c.w[k * (Ms + 1) + Ms] = ymean[k];
        }
        const std::vector<double> p = readout_predict(c, F, s.val0, s.val1);
        if (classify) {
            const std::vector<int> cls = readout_argmax(p, K);
            size_t right = 0;
            for (size_t t = 0; t < cls.size(); ++t) right += Y[(s.val0 + t) * Ks + (size_t)cls[t]] > 0.5;
            const double acc = (double)right / (double)cls.size();
            if (acc >= r.val_accuracy) {
                r.val_accuracy = acc;
                r.w = c.w;
                for (size_t k = 0; k < Ks; ++k) {
                    r.lambda[k] = lam;
                    r.val_nrmse[k] = readout_nrmse(Y.data() + s.val0 * Ks + k, p.data() + k, Ks, s.val1 - s.val0);
                }
            }
            continue;
        }
        for (size_t k = 0; k < Ks; ++k) {
            const double e = readout_nrmse(Y.data() + s.val0 * Ks + k, p.data() + k, Ks, s.val1 - s.val0);
            if (e < r.val_nrmse[k]) {
                r.val_nrmse[k] = e;
                r.lambda[k] = lam;
                std::copy(c.w.begin() + (std::ptrdiff_t)(k * (Ms + 1)), c.w.begin() + (std::ptrdiff_t)((k + 1) * (Ms + 1)), r.w.begin() + (std::ptrdiff_t)(k * (Ms + 1)));
            }
        }
    }
    return r;
}
