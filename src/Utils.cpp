/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
    Copyright (C) 2018-2019 SAI Team

    SAI is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SAI is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SAI.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/


#include "config.h"
#include "Utils.h"

#include <algorithm>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cmath>

#include <boost/filesystem.hpp>
#include <boost/math/distributions/students_t.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/select.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#endif

#include "GTP.h"

Utils::ThreadPool thread_pool;

auto constexpr z_entries = 1000;
std::array<float, z_entries> z_lookup;

void Utils::create_z_table() {
    for (auto i = 1; i < z_entries + 1; i++) {
        boost::math::students_t dist(i);
        auto z = boost::math::quantile(boost::math::complement(dist, cfg_ci_alpha));
        z_lookup[i - 1] = z;
    }
}

float Utils::cached_t_quantile(int v) {
    if (v < 1) {
        return z_lookup[0];
    }
    if (v < z_entries) {
        return z_lookup[v - 1];
    }
    // z approaches constant when v is high enough.
    // With default lookup table size the function is flat enough that we
    // can just return the last entry for all v bigger than it.
    return z_lookup[z_entries - 1];
}

bool Utils::input_pending() {
#ifdef HAVE_SELECT
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(0,&read_fds);
    struct timeval timeout{0,0};
    select(1,&read_fds,nullptr,nullptr,&timeout);
    return FD_ISSET(0, &read_fds);
#else
    static int init = 0, pipe;
    static HANDLE inh;
    DWORD dw;

    if (!init) {
        init = 1;
        inh = GetStdHandle(STD_INPUT_HANDLE);
        pipe = !GetConsoleMode(inh, &dw);
        if (!pipe) {
            SetConsoleMode(inh, dw & ~(ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT));
            FlushConsoleInputBuffer(inh);
        }
    }

    if (pipe) {
        if (!PeekNamedPipe(inh, nullptr, 0, nullptr, &dw, nullptr)) {
            myprintf("Nothing at other end - exiting\n");
            exit(EXIT_FAILURE);
        }

        return dw;
    } else {
        if (!GetNumberOfConsoleInputEvents(inh, &dw)) {
            myprintf("Nothing at other end - exiting\n");
            exit(EXIT_FAILURE);
        }

        return dw > 1;
    }
    return false;
#endif
}

static std::mutex IOmutex;

static void myprintf_base(const char *fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);

    vfprintf(stderr, fmt, ap);

    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        vfprintf(cfg_logfile_handle, fmt, ap2);
    }
    va_end(ap2);
}

void Utils::myprintf(const char *fmt, ...) {
    if (cfg_quiet) {
        return;
    }

    va_list ap;
    va_start(ap, fmt);
    myprintf_base(fmt, ap);
    va_end(ap);
}

void Utils::myprintf_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    myprintf_base(fmt, ap);
    va_end(ap);
}

static void gtp_fprintf(FILE* file, const std::string& prefix,
                        const char *fmt, va_list ap) {
    fprintf(file, "%s ", prefix.c_str());
    vfprintf(file, fmt, ap);
    fprintf(file, "\n\n");
}

static void gtp_base_printf(int id, std::string prefix,
                            const char *fmt, va_list ap) {
    if (id != -1) {
        prefix += std::to_string(id);
    }
    gtp_fprintf(stdout, prefix, fmt, ap);
    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        gtp_fprintf(cfg_logfile_handle, prefix, fmt, ap);
    }
}

void Utils::gtp_printf(int id, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    gtp_base_printf(id, "=", fmt, ap);
    va_end(ap);
}

void Utils::gtp_printf_raw(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);

    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        va_start(ap, fmt);
        vfprintf(cfg_logfile_handle, fmt, ap);
        va_end(ap);
    }
}

void Utils::gtp_fail_printf(int id, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    gtp_base_printf(id, "?", fmt, ap);
    va_end(ap);
}

void Utils::log_input(const std::string& input) {
    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        fprintf(cfg_logfile_handle, ">>%s\n", input.c_str());
    }
}

size_t Utils::ceilMultiple(size_t a, size_t b) {
    if (a % b == 0) {
        return a;
    }

    auto ret = a + (b - a % b);
    return ret;
}

float Utils::sigmoid_interval_avg(float alpkt, float beta, float beta2, float s, float t) {
    const auto h = t - s;

    if (std::abs(h) < 0.001f) {
        return sigmoid(alpkt,beta,(s+t)/2.0f,beta2).first;
    }

    #ifndef NDEBUG
    if (std::abs(s) + std::abs(t) > 2000.0f) {
        myprintf("Warning: integration interval out of bound: [%f,%f].\n", s, t);
    }
    #endif

    const auto a = std::abs(alpkt+s);
    const auto b = std::abs(alpkt+t);

    const double main_term = (alpkt+s)*(alpkt+t) > 0 ?
        ( alpkt+s > 0 ? 1.0 : 0.0 ) :
        0.5 + 0.5 * (b-a) / h;

    if (beta2 < 0) {
        const auto aa = log_sigmoid(a*beta);
        const auto bb = log_sigmoid(b*beta);

        return float(main_term + (aa - bb) / h / beta);
    } else {
        const auto cf_a = alpkt+s < 0 ? 1.0 / beta : 1.0 / beta2;
        const auto cf_b = alpkt+t < 0 ? 1.0 / beta : 1.0 / beta2;
        constexpr double log2 = 0.69314718055994530941723212145818;
        const double delta_term = (alpkt+s)*(alpkt+t) > 0 ?
                                    0.0 : (cf_a - cf_b) / h * log2;

        const auto aa = log_sigmoid(a / cf_a) * cf_a;
        const auto bb = log_sigmoid(b / cf_b) * cf_b;

        return float(main_term + delta_term + (aa - bb) / h);
    }
}

double Utils::log_sigmoid(double x) {
    // Returns log(1/(1+exp(-x))
    // When sigmoid is about 1, log(sigmoid) is about 0 but not very precise
    // This function provides a robust estimate in that case.
    // Its argument should be beta*(alpha+bonus) and be positive.
    if (x > 10.0) {
        const auto ul = std::exp(-x);
        const auto ll = 1.0 / (1.0 + std::exp(x));
        return -std::sqrt(ul * ll);
    }
    return std::log(double(sigmoid(float(x), 1.0, 0.0).first));
}

float Utils::median(std::vector<float> & sample){
    sort(sample.begin(), sample.end());
    const auto len = sample.size();
    float result;
    if(len % 2) {
        result = sample[len/2];
    } else {
        result = 0.5 * sample[len/2] + 0.5 * sample[len/2-1];
    }
    return result;
}


float Utils::winner(float board_score) {
    if (board_score > 0.0001f) {
        return 1.0f;
    }

    if (board_score < -0.0001f) {
        return 0.0f;
    }

    return 0.5f;
}

const std::string Utils::leelaz_file(std::string file) {
#if defined(_WIN32) || defined(__ANDROID__)
    boost::filesystem::path dir(boost::filesystem::current_path());
#else
    // https://stackoverflow.com/a/26696759
    const char *homedir;
    if ((homedir = getenv("HOME")) == nullptr) {
        struct passwd *pwd;
        if ((pwd = getpwuid(getuid())) == nullptr) { // NOLINT(runtime/threadsafe_fn)
            return std::string();
        }
        homedir = pwd->pw_dir;
    }
    boost::filesystem::path dir(homedir);
    dir /= ".local/share/sai";
#endif
    boost::filesystem::create_directories(dir);
    dir /= file;
    return dir.string();
}

bool Utils::parse_agent_params(std::array<float, 4> &params, const std::string &str) {
    std::vector<float> numbers;
    std::stringstream ss(str);
    while (ss.good()) {
        std::string substr;
        std::getline(ss, substr, ',');
        numbers.push_back(std::stof(substr));
    }
    const auto n = numbers.size();
    if (n>4) {
        return false;
    }

    switch(n) {
    case 1:
        // CPU when behind = CPU when ahead
        numbers.push_back(numbers[0]);
        // fall through
    case 2:
        // human when ahead = CPU when ahead
        numbers.push_back(numbers[0]);
        // fall through
    case 3:
        // human when behind = CPU when behind
        numbers.push_back(numbers[1]);
    }
    std::copy(cbegin(numbers), cend(numbers), begin(params));

    return true;
}

void Utils::dump_agent_params() {
    std::array<std::string, 4> labels = {"Agent parameters (lambda, mu):"
                                         "\n CPU,   ahead",
                                        ", behind",
                                         "\n human, ahead",
                                        ", behind"};
    for (auto i = 0 ; i<4 ; i++) {
        myprintf("%s (%.2f, %.2f)", labels[i].c_str(), cfg_lambda[i], cfg_mu[i]);
    }
    myprintf("\n");

}

bool Utils::agent_color_dependent() {
    auto dependent = false;
    dependent |= (cfg_lambda[0] != cfg_lambda[2]);
    dependent |= (cfg_lambda[1] != cfg_lambda[3]);
    dependent |= (cfg_mu[0] != cfg_mu[2]);
    dependent |= (cfg_mu[1] != cfg_mu[3]);

    return dependent;
}
