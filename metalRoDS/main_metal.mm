/*
 * main_metal.mm
 * Metal (Apple GPU) translation of the CUDA DEM rod-particle simulation.
 * Physics is identical to the original full_code_main.cu.
 *
 * IMPORTANT: float3 and int3 are defined here BEFORE importing Metal headers
 * to avoid conflicts with simd types.
 */

// ============================================================
//  float3 / int3 definitions  (MUST come before Metal headers)
// ============================================================
struct float3 { float x, y, z; };
struct int3   { int   x, y, z; };

inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline int3   make_int3  (int   x, int   y, int   z) { return {x, y, z}; }

// ============================================================
//  C++ standard includes
// ============================================================
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <string>
#include <vector>
#include <thread>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <utility>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <typeinfo>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <array>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <nlohmann/json.hpp>

// ============================================================
//  Metal / Foundation headers
// ============================================================
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// ============================================================
//  misc_funcs  (CPU versions — CUDA qualifiers stripped)
// ============================================================

const float h_PI = 3.14159265358979323846f;

class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    std::string time_elapsed() {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        long long totalMs = duration.count();
        long long days  = totalMs / 86400000;
        long long hours = (totalMs % 86400000) / 3600000;
        long long mins  = (totalMs % 3600000)  / 60000;
        long long secs  = (totalMs % 60000)    / 1000;
        long long ms    = totalMs % 1000;
        std::vector<std::string> parts;
        if (days  > 0) parts.push_back(std::to_string(days)  + "d");
        if (hours > 0) parts.push_back(std::to_string(hours) + "h");
        if (mins  > 0) parts.push_back(std::to_string(mins)  + "m");
        if (secs  > 0) parts.push_back(std::to_string(secs)  + "s");
        if (ms    > 0) parts.push_back(std::to_string(ms)    + "ms");
        std::string result;
        for (size_t i = 0; i < parts.size(); ++i) {
            if (i > 0) result += ' ';
            result += parts[i];
        }
        return result;
    }
    double elapsed_seconds() {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        return duration.count() * 1e-3;
    }
    void pause(int seconds) { std::this_thread::sleep_for(std::chrono::seconds(seconds)); }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time, stop_time;
};

class Helper {
public:
    bool  doPrint;
    float total_strain;
    float total_time;
    bool  jammed_status;
    float shear_rate;
};

struct OutputFlags { bool doFluidOut; bool doVisOut; bool doSimpleOut; bool doCheckpointOut; };

struct bool3 { bool x, y, z; };

struct mat33 {
    float m[9];
    mat33() : m{} {}
};

// float3 operators
inline float3 operator+(const float3 &a, const float3 &b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline float3 operator-(const float3 &a, const float3 &b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline float3 operator*(const float3 &a, float s)         { return {a.x*s, a.y*s, a.z*s}; }
inline float3 operator*(float s, const float3 &a)         { return {a.x*s, a.y*s, a.z*s}; }
inline float3 operator/(const float3 &a, float s)         { return {a.x/s, a.y/s, a.z/s}; }
inline float3& operator+=(float3 &a, const float3 &b)     { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
inline float3 elementwise_mul(const float3 &A, const float3 &B) { return {A.x*B.x, A.y*B.y, A.z*B.z}; }
inline float mag(const float3 &v) { return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }
inline float dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float3 cross(float3 a, float3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline float signf(float x) { return x >= 0.0f ? 1.0f : -1.0f; }
inline float3 normalize(const float3 &v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len > 0.0f) return {v.x/len, v.y/len, v.z/len};
    return {0.0f, 0.0f, 0.0f};
}
inline float clampf(float v, float lo, float hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }

float rand01()       { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); }
float rand_m1_to_1() { return 2.0f * rand01() - 1.0f; }

inline float mod(float a, float b) { return std::fmod(std::fmod(a,b)+b, b); }

inline int int_index(int i, int j, int N) {
    return i * N - (i * (i - 1)) / 2 + (j - 1);
}

// mat33 operations
inline mat33 elementmat_mul(const mat33 &A, const mat33 &B) {
    mat33 R;
    for (int i = 0; i < 9; ++i) R.m[i] = A.m[i] * B.m[i];
    return R;
}
inline mat33 elementmat_div(const mat33 &A, const mat33 &B) {
    mat33 R;
    for (int i = 0; i < 9; ++i) R.m[i] = A.m[i] / B.m[i];
    return R;
}
inline mat33 inverse_mat(const mat33 &A) {
    mat33 inv{};
    float det = A.m[0]*(A.m[4]*A.m[8]-A.m[5]*A.m[7])
               -A.m[1]*(A.m[3]*A.m[8]-A.m[5]*A.m[6])
               +A.m[2]*(A.m[3]*A.m[7]-A.m[4]*A.m[6]);
    if (fabsf(det) < 1e-12f) return inv;
    float invDet = 1.0f / det;
    inv.m[0] =  (A.m[4]*A.m[8]-A.m[5]*A.m[7])*invDet;
    inv.m[1] = -(A.m[1]*A.m[8]-A.m[2]*A.m[7])*invDet;
    inv.m[2] =  (A.m[1]*A.m[5]-A.m[2]*A.m[4])*invDet;
    inv.m[3] = -(A.m[3]*A.m[8]-A.m[5]*A.m[6])*invDet;
    inv.m[4] =  (A.m[0]*A.m[8]-A.m[2]*A.m[6])*invDet;
    inv.m[5] = -(A.m[0]*A.m[5]-A.m[2]*A.m[3])*invDet;
    inv.m[6] =  (A.m[3]*A.m[7]-A.m[4]*A.m[6])*invDet;
    inv.m[7] = -(A.m[0]*A.m[7]-A.m[1]*A.m[6])*invDet;
    inv.m[8] =  (A.m[0]*A.m[4]-A.m[1]*A.m[3])*invDet;
    return inv;
}
inline mat33 mat_scal_mul(const mat33 &A, float s) {
    mat33 R;
    for (int i = 0; i < 9; ++i) R.m[i] = A.m[i] * s;
    return R;
}
inline mat33 test_mat_mul(const mat33 &A, const mat33 &B) {
    mat33 R;
    R.m[0]=A.m[0]*B.m[0]+A.m[1]*B.m[3]+A.m[2]*B.m[6];
    R.m[1]=A.m[0]*B.m[1]+A.m[1]*B.m[4]+A.m[2]*B.m[7];
    R.m[2]=A.m[0]*B.m[2]+A.m[1]*B.m[5]+A.m[2]*B.m[8];
    R.m[3]=A.m[3]*B.m[0]+A.m[4]*B.m[3]+A.m[5]*B.m[6];
    R.m[4]=A.m[3]*B.m[1]+A.m[4]*B.m[4]+A.m[5]*B.m[7];
    R.m[5]=A.m[3]*B.m[2]+A.m[4]*B.m[5]+A.m[5]*B.m[8];
    R.m[6]=A.m[6]*B.m[0]+A.m[7]*B.m[3]+A.m[8]*B.m[6];
    R.m[7]=A.m[6]*B.m[1]+A.m[7]*B.m[4]+A.m[8]*B.m[7];
    R.m[8]=A.m[6]*B.m[2]+A.m[7]*B.m[5]+A.m[8]*B.m[8];
    return R;
}
inline mat33 outer_product(const float3 &a, const float3 &b) {
    mat33 result;
    result.m[0]=a.x*b.x; result.m[1]=a.x*b.y; result.m[2]=a.x*b.z;
    result.m[3]=a.y*b.x; result.m[4]=a.y*b.y; result.m[5]=a.y*b.z;
    result.m[6]=a.z*b.x; result.m[7]=a.z*b.y; result.m[8]=a.z*b.z;
    return result;
}

void fluid_stress(
    float *vol_frac_profile, const float &aspect, const float &viscosity,
    float *gradient_profile, const float &num_bins, float *fluid_stress_profile)
{
    for (int i = 0; i < (int)num_bins; ++i) {
        float pure_fluid_part = viscosity * gradient_profile[i];
        float einstein = 1.0f + 2.5f*vol_frac_profile[i] + vol_frac_profile[i]/16.0f*aspect*aspect;
        fluid_stress_profile[i] = pure_fluid_part * einstein;
    }
}

void shear_stress(
    const float &num_bins,
    const float *cn_stress_profile, const float *ct_stress_profile,
    const float *l_stress_profile,  const float *fluid_stress_profile,
    float *shear_stress_profile)
{
    for (int i = 0; i < (int)num_bins; ++i)
        shear_stress_profile[i] = cn_stress_profile[i] + ct_stress_profile[i]
                                 + l_stress_profile[i] + fluid_stress_profile[i];
}

void stress_controller(
    float *fixed_stress_profile, float *stress_profile,
    float *gradient_profile, int num_bins, float controller_gain)
{
    for (int i = 0; i < num_bins; ++i) {
        float error = (fixed_stress_profile[i] - stress_profile[i]) / fixed_stress_profile[i];
        gradient_profile[i] = fmaxf(0.0f, gradient_profile[i] + controller_gain * error);
    }
}

double S_func(double w) {
    double term1 = 3.0*(w*w+(1.0-w)*(1.0-w))/(8.0*w*w);
    double log_term = std::abs(2.0*w-1.0);
    if (log_term == 0.0) log_term = 1e-12;
    double term2 = (3.0*std::pow(2.0*w-1.0,2))/(16.0*std::pow(w,3)*(1.0-w));
    term2 *= std::log(log_term);
    return 1.0 - term1 - term2;
}

double find_w_for_S(double target_S, double tol=1e-8, int max_iter=1000) {
    double low=0.001, high=0.999, mid=0.5;
    for (int i=0; i<max_iter; ++i) {
        mid = 0.5*(low+high);
        double S_mid = S_func(mid);
        if (std::abs(S_mid-target_S)<tol) return mid;
        if (S_mid < target_S) low = mid; else high = mid;
    }
    return mid;
}

Eigen::Vector3f float3_to_eigenf(const float3 &u) { return Eigen::Vector3f(u.x, u.y, u.z); }
float3 eigenf_to_float3(const Eigen::Vector3f &u) { return {u.x(), u.y(), u.z()}; }

Eigen::Matrix3f calc_q_tensor(const int num_rods, const float3 *oris) {
    Eigen::Matrix3f Q = Eigen::Matrix3f::Zero();
    const Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    for (int i = 0; i < num_rods; ++i) {
        Eigen::Vector3f u = float3_to_eigenf(oris[i]);
        Q += 1.5f * (u * u.transpose() - (1.0f/3.0f) * I);
    }
    Q /= static_cast<float>(num_rods);
    return Q;
}

float calc_oop_ref_v1(Eigen::Matrix3f Q, const float3 n_ref) {
    Eigen::Vector3f n = float3_to_eigenf(n_ref).normalized();
    return n.dot(Q * n);
}

std::pair<float, float3> calc_oop_and_director(Eigen::Matrix3f Q) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(Q);
    Eigen::Vector3f evals = solver.eigenvalues();
    Eigen::Matrix3f evecs = solver.eigenvectors();
    int idx{};
    float S = evals.maxCoeff(&idx);
    Eigen::Vector3f director_vec = evecs.col(idx).normalized();
    return {S, eigenf_to_float3(director_vec)};
}

void convert_to_avg_viscosity(
    float *stress_profile, float fluid_viscosity,
    float *gradient_profile, int num_bins, float &avg_visc)
{
    for (int i = 0; i < num_bins; ++i)
        avg_visc += stress_profile[i] / (fluid_viscosity * gradient_profile[i]);
    avg_visc /= num_bins;
}

void avg_array(float *array, float &avg_prop, int num_bins) {
    for (int i = 0; i < num_bins; ++i) avg_prop += array[i];
    avg_prop /= num_bins;
}

void shift_and_append(float arr[], int N, float new_value) {
    for (int i = 0; i < N-1; i++) arr[i] = arr[i+1];
    arr[N-1] = new_value;
}

std::pair<float,float> fit_to_sin(const std::vector<float>& data, const std::vector<float>& time, float frequency) {
    int n = (int)data.size();
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    double sTs=0.0, sTc=0.0, cTc=0.0, sTy=0.0, cTy=0.0;
    for (int i=0; i<n; ++i) {
        double y = data[i] - mean;
        double angle = frequency * time[i];
        double s = std::sin(angle), c = std::cos(angle);
        sTs += s*s; cTc += c*c; sTc += s*c; sTy += s*y; cTy += c*y;
    }
    double det = sTs*cTc - sTc*sTc;
    double a = (cTc*sTy - sTc*cTy)/det;
    double b = (-sTc*sTy + sTs*cTy)/det;
    float A = static_cast<float>(std::hypot(a,b));
    float delta = static_cast<float>(std::atan2(b,a));
    return {A, delta};
}

float average_property(const float* property, int num_particles) {
    if (num_particles <= 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < num_particles; ++i) sum += property[i];
    return sum / num_particles;
}

// ============================================================
//  Velocity profile functions  (CPU versions)
// ============================================================

void velocity_gradient_profile_eval(
    const float *velocity_profile, int num_velos_specified,
    float bin_size, float *gradient_profile)
{
    if (num_velos_specified < 1) { printf("Need to specify a fluid velocity\n"); }
    if (num_velos_specified < 2) { gradient_profile[0] = 0.0f; return; }
    for (int i = 0; i < num_velos_specified-1; ++i)
        gradient_profile[i] = (velocity_profile[i+1] - velocity_profile[i]) / bin_size;
}

void velocity_profile_eval(float *velocity_profile, int num_bins, float bin_size, float *gradient_profile) {
    velocity_profile[0] = 0.0f;
    for (int i = 0; i < num_bins; ++i)
        velocity_profile[i+1] = velocity_profile[i] + gradient_profile[i] * bin_size;
}

// ============================================================
//  input_funcs
// ============================================================

struct Float3 { float x, y, z; };

std::unordered_map<std::string,std::string> loadConfig(const std::string &path) {
    std::unordered_map<std::string,std::string> out;
    std::ifstream ifs(path);
    if (!ifs) return out;
    std::string line;
    while (std::getline(ifs, line)) {
        auto posc = line.find('#');
        if (posc != std::string::npos) line = line.substr(0, posc);
        auto l = line.find_first_not_of(" \t\r\n");
        if (l == std::string::npos) continue;
        auto r = line.find_last_not_of(" \t\r\n");
        std::string trimmed = line.substr(l, r-l+1);
        if (trimmed.empty()) continue;
        auto eq = trimmed.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trimmed.substr(0, eq);
        std::string val = trimmed.substr(eq+1);
        auto trim = [](std::string &s) {
            auto l = s.find_first_not_of(" \t\r\n");
            auto r = s.find_last_not_of(" \t\r\n");
            if (l == std::string::npos) { s.clear(); return; }
            s = s.substr(l, r-l+1);
        };
        trim(key); trim(val);
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c){ return std::tolower(c); });
        out[key] = val;
    }
    return out;
}

bool toBool(const std::string &s) {
    std::string t; t.reserve(s.size());
    for (char c : s) t.push_back(std::tolower((unsigned char)c));
    return t=="1"||t=="true"||t=="True"||t=="yes"||t=="on";
}
int   toInt  (const std::string &s) { return std::stoi(s); }
float toFloat(const std::string &s) { return std::stof(s); }
long long toLongLong(const std::string &s) { return std::stoll(s); }
double toDouble(const std::string &s) { return std::stod(s); }

std::vector<double> parseVectorFloat(const std::string &s) {
    std::string t;
    for (char c : s) if (c!='['&&c!=']'&&c!='('&&c!=')') t.push_back(c);
    for (char &c : t) if (c==',') c=' ';
    std::istringstream iss(t);
    std::vector<double> out; double v;
    while (iss>>v) out.push_back(v);
    return out;
}

std::vector<int> parseIntVector(const std::string &s) {
    std::string t;
    for (char c : s) if (c!='['&&c!=']'&&c!='('&&c!=')') t.push_back(c);
    for (char &c : t) if (c==',') c=' ';
    std::istringstream iss(t);
    std::vector<int> out; int x;
    while (iss>>x) out.push_back(x);
    return out;
}

std::vector<bool> parseBoolVector(const std::string &s) {
    auto ints = parseIntVector(s);
    if (!ints.empty()) {
        std::vector<bool> out; out.reserve(ints.size());
        for (int i: ints) out.push_back(i!=0);
        return out;
    }
    std::vector<bool> out;
    std::vector<double> dv = parseVectorFloat(s);
    out.reserve(dv.size());
    for (double d : dv) out.push_back(d!=0.0);
    return out;
}

Float3 parseVector3f(const std::string &s) {
    auto v = parseVectorFloat(s);
    Float3 r{0.f,0.f,0.f};
    if (v.size()>=1) r.x = static_cast<float>(v[0]);
    if (v.size()>=2) r.y = static_cast<float>(v[1]);
    if (v.size()>=3) r.z = static_cast<float>(v[2]);
    return r;
}

bool get_box_size_simple(const std::string &path, std::array<double,3> &L) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("ITEM: BOX BOUNDS") != std::string::npos) {
            for (int i = 0; i < 3; ++i) {
                if (!std::getline(f, line)) return false;
                std::istringstream iss(line);
                double lo, hi;
                if (!(iss >> lo >> hi)) return false;
                L[i] = hi - lo;
            }
            return true;
        }
    }
    return false;
}

int numrods_polyfile(const std::string &particle_list_file) {
    std::ifstream ifs(particle_list_file);
    std::string line;
    int count = 0;
    while (std::getline(ifs, line)) {
        auto first_non = line.find_first_not_of(" \t\r\n");
        if (first_non == std::string::npos) continue;
        line = line.substr(first_non);
        if (line.rfind("vol_frac",0)==0) continue;
        if (line.find(',') != std::string::npos) ++count;
    }
    return count;
}

// ============================================================
//  export_funcs
// ============================================================

std::string create_output_folder(std::string &save_name, std::string &file_path) {
    std::string folder_path = file_path + save_name + "/";
    if (!std::filesystem::exists(folder_path))
        if (!std::filesystem::create_directory(folder_path))
            std::cout << "failed to create output folder\n";
    return folder_path;
}

std::string create_output_subfolder(std::string folder_name, std::string &folder_path) {
    std::string subfolder_path = folder_path + folder_name + "/";
    if (!std::filesystem::exists(subfolder_path))
        if (!std::filesystem::create_directory(subfolder_path))
            std::cout << "failed to create output subfolder\n";
    return subfolder_path;
}

struct Quaternion { float x, y, z, w; };

Quaternion quaternionFromOri(const float3 &ori) {
    const float3 ref = {0.0f, 0.0f, 1.0f};
    float3 diff = ori - ref;
    if (sqrtf(dot(diff,diff)) < 1e-6f) return {0.0f,0.0f,0.0f,1.0f};
    float3 sum = ori + ref;
    if (sqrtf(dot(sum,sum)) < 1e-6f) return {1.0f,0.0f,0.0f,0.0f};
    float3 axis = normalize(cross(ref, ori));
    float cosTheta = clampf(dot(ref, ori), -1.0f, 1.0f);
    float theta = acosf(cosTheta);
    float half = 0.5f * theta;
    float s = sinf(half), c = cosf(half);
    return {axis.x*s, axis.y*s, axis.z*s, c};
}

void to_lammps_dump(
    int &num_particles, int &time_step, float3 &system_dimensions,
    float3 *CoMs, float3 *oris, float *radii, float *shaft_lengths,
    float3 *velos, float3 *avelos, float3 *forces, float3 *torques,
    mat33 *cn_stresses, mat33 *ct_stresses, mat33 *l_stresses,
    int *coord_nums, std::string &folder_path)
{
    std::string filename = folder_path + "dump.rod";
    if (time_step == 0) std::remove(filename.c_str());
    std::ofstream file(filename, std::ios::app);
    if (!file) { std::cerr << "Error opening dump file: " << filename << "\n"; return; }
    file << "ITEM: TIMESTEP\n" << time_step << '\n';
    file << "ITEM: NUMBER OF ATOMS\n" << num_particles << '\n';
    file << "ITEM: BOX BOUNDS pp pp pp\n";
    file << "0 " << system_dimensions.x << '\n'
         << "0 " << system_dimensions.y << '\n'
         << "0 " << system_dimensions.z << '\n';
    file << "ITEM: ATOMS id x y z quatw quati quatj quatk shapex shapey shapez unit_x unit_y unit_z coord_num vx vy vz avx avy avz fx fy fz tx ty tz sxz cnsxz ctsxz lsxz\n";
    for (int i = 0; i < num_particles; ++i) {
        Quaternion q = quaternionFromOri(oris[i]);
        float cnsxz = (cn_stresses[i].m[2] + cn_stresses[i].m[6]) / 2.0f;
        float ctsxz = (ct_stresses[i].m[2] + ct_stresses[i].m[6]) / 2.0f;
        float lsxz  = (l_stresses[i].m[2]  + l_stresses[i].m[6])  / 2.0f;
        float sxz   = cnsxz + ctsxz + lsxz;
        file << i << " "
             << CoMs[i].x << " " << CoMs[i].y << " " << CoMs[i].z << " "
             << q.w << " " << q.x << " " << q.y << " " << q.z << " "
             << radii[i]*2.0f << " " << radii[i]*2.0f << " " << shaft_lengths[i]*2.0f << " "
             << oris[i].x << " " << oris[i].y << " " << oris[i].z << " "
             << coord_nums[i] << " "
             << velos[i].x << " " << velos[i].y << " " << velos[i].z << " "
             << avelos[i].x << " " << avelos[i].y << " " << avelos[i].z << " "
             << forces[i].x << " " << forces[i].y << " " << forces[i].z << " "
             << torques[i].x << " " << torques[i].y << " " << torques[i].z << " "
             << sxz << " " << cnsxz << " " << ctsxz << " " << lsxz << '\n';
    }
    file.close();
}

inline bool write_bytes(std::ofstream &f, const void* data, std::size_t bytes) {
    f.write(reinterpret_cast<const char*>(data), bytes); return bool(f);
}
inline bool read_bytes(std::ifstream &f, void* data, std::size_t bytes) {
    f.read(reinterpret_cast<char*>(data), bytes); return bool(f);
}

bool write_particles_binary(
    const std::string &fname, int num_particles, float LEBC_shift,
    const int *ids, const float *radii, const float *aspects, const float *shaft_lengths,
    const float *densities, const float *volumes, const float *masses,
    const float3 *moments_of_inertia, const float *avg_inertias,
    const float *kns, const float *ens, const float *ets, const float *fric_coefs,
    const float3 *CoMs, const float3 *oris, const float3 *endpoints1, const float3 *endpoints2,
    const float3 *tvelos, const float3 *avelos, const float3 *accelerations, const float3 *angular_accelerations,
    const float3 *forces, const float3 *torques,
    const float3 *old_interactions, const float3 *new_interactions, long long num_interactions)
{
    std::ofstream f(fname, std::ios::binary);
    if (!f) { std::cerr << "write_particles_binary: cannot open " << fname << "\n"; return false; }
    uint32_t magic=0x50415254u, version=1;
    write_bytes(f,&magic,sizeof(magic)); write_bytes(f,&version,sizeof(version));
    write_bytes(f,&num_particles,sizeof(num_particles));
    write_bytes(f,&num_interactions,sizeof(num_interactions));
    write_bytes(f,&LEBC_shift,sizeof(LEBC_shift));
    write_bytes(f,ids,sizeof(int)*(size_t)num_particles);
    write_bytes(f,radii,sizeof(float)*(size_t)num_particles);
    write_bytes(f,aspects,sizeof(float)*(size_t)num_particles);
    write_bytes(f,shaft_lengths,sizeof(float)*(size_t)num_particles);
    write_bytes(f,densities,sizeof(float)*(size_t)num_particles);
    write_bytes(f,volumes,sizeof(float)*(size_t)num_particles);
    write_bytes(f,masses,sizeof(float)*(size_t)num_particles);
    for (int i=0;i<num_particles;++i) {
        float trip[3]={moments_of_inertia[i].x,moments_of_inertia[i].y,moments_of_inertia[i].z};
        write_bytes(f,trip,sizeof(trip));
    }
    write_bytes(f,avg_inertias,sizeof(float)*(size_t)num_particles);
    write_bytes(f,kns,sizeof(float)*(size_t)num_particles);
    write_bytes(f,ens,sizeof(float)*(size_t)num_particles);
    write_bytes(f,ets,sizeof(float)*(size_t)num_particles);
    write_bytes(f,fric_coefs,sizeof(float)*(size_t)num_particles);
    auto wf3=[&](const float3 *arr){ for(int i=0;i<num_particles;++i){float t[3]={arr[i].x,arr[i].y,arr[i].z};write_bytes(f,t,sizeof(t));} };
    wf3(CoMs);wf3(oris);wf3(endpoints1);wf3(endpoints2);
    wf3(tvelos);wf3(avelos);wf3(forces);wf3(torques);wf3(accelerations);wf3(angular_accelerations);
    for(long long i=0;i<num_interactions;++i){float t[3]={old_interactions[i].x,old_interactions[i].y,old_interactions[i].z};write_bytes(f,t,sizeof(t));}
    for(long long i=0;i<num_interactions;++i){float t[3]={new_interactions[i].x,new_interactions[i].y,new_interactions[i].z};write_bytes(f,t,sizeof(t));}
    f.close(); return true;
}

bool read_particles_header(const std::string &fname, int &out_num_particles, long long &out_num_interactions, float &LEBC_shift) {
    std::ifstream f(fname, std::ios::binary);
    if (!f) { std::cerr << "read_particles_header: cannot open " << fname << "\n"; return false; }
    uint32_t magic=0, version=0;
    if (!read_bytes(f,&magic,sizeof(magic))) return false;
    if (magic!=0x50415254u) { std::cerr << "bad file magic\n"; return false; }
    read_bytes(f,&version,sizeof(version));
    read_bytes(f,&out_num_particles,sizeof(out_num_particles));
    read_bytes(f,&out_num_interactions,sizeof(out_num_interactions));
    read_bytes(f,&LEBC_shift,sizeof(LEBC_shift));
    f.close(); return true;
}

bool read_particles_binary_into_arrays(
    const std::string &fname, int expected_num_particles,
    int *ids, float *radii, float *aspects, float *shaft_lengths,
    float *densities, float *volumes, float *masses,
    float3 *moments_of_inertia, float *avg_inertias,
    float *kns, float *ens, float *ets, float *fric_coefs,
    float3 *CoMs, float3 *oris, float3 *endpoints1, float3 *endpoints2,
    float3 *tvelos, float3 *avelos,
    float3 *forces, float3 *torques, float3 *accelerations, float3 *angular_accelerations,
    float3 *old_interactions, float3 *new_interactions,
    long long expected_num_interactions)
{
    std::ifstream f(fname, std::ios::binary);
    if (!f) { std::cerr << "read_particles_binary_into_arrays: cannot open " << fname << "\n"; return false; }
    uint32_t magic=0,version=0; int np=0; long long ni=0; float lebc=0;
    if (!read_bytes(f,&magic,sizeof(magic))) return false;
    if (magic!=0x50415254u) { std::cerr << "bad file magic\n"; return false; }
    read_bytes(f,&version,sizeof(version));
    read_bytes(f,&np,sizeof(np));
    read_bytes(f,&ni,sizeof(ni));
    read_bytes(f,&lebc,sizeof(lebc));
    if (np!=expected_num_particles){std::cerr<<"read: expected "<<expected_num_particles<<" but file has "<<np<<"\n";return false;}
    if (ni!=expected_num_interactions){std::cerr<<"read: expected interactions "<<expected_num_interactions<<" but file has "<<ni<<"\n";return false;}
    read_bytes(f,ids,sizeof(int)*(size_t)np);
    read_bytes(f,radii,sizeof(float)*(size_t)np);
    read_bytes(f,aspects,sizeof(float)*(size_t)np);
    read_bytes(f,shaft_lengths,sizeof(float)*(size_t)np);
    read_bytes(f,densities,sizeof(float)*(size_t)np);
    read_bytes(f,volumes,sizeof(float)*(size_t)np);
    read_bytes(f,masses,sizeof(float)*(size_t)np);
    for(int i=0;i<np;++i){float trip[3];read_bytes(f,trip,sizeof(trip));moments_of_inertia[i]={trip[0],trip[1],trip[2]};}
    read_bytes(f,avg_inertias,sizeof(float)*(size_t)np);
    read_bytes(f,kns,sizeof(float)*(size_t)np);
    read_bytes(f,ens,sizeof(float)*(size_t)np);
    read_bytes(f,ets,sizeof(float)*(size_t)np);
    read_bytes(f,fric_coefs,sizeof(float)*(size_t)np);
    auto rf3=[&](float3 *arr){for(int i=0;i<np;++i){float t[3];read_bytes(f,t,sizeof(t));arr[i]={t[0],t[1],t[2]};}};
    rf3(CoMs);rf3(oris);rf3(endpoints1);rf3(endpoints2);
    rf3(tvelos);rf3(avelos);rf3(forces);rf3(torques);rf3(accelerations);rf3(angular_accelerations);
    for(long long i=0;i<ni;++i){float t[3];read_bytes(f,t,sizeof(t));old_interactions[i]={t[0],t[1],t[2]};}
    for(long long i=0;i<ni;++i){float t[3];read_bytes(f,t,sizeof(t));new_interactions[i]={t[0],t[1],t[2]};}
    f.close(); return true;
}

void fluid_info_output(
    int frame, float strain, float total_time, float dt, float dt_avg,
    float fluid_viscosity, float fluid_density, float max_height, int num_bins,
    float *velocity_profile, float *gradient_profile,
    float *total_stress_profile, float *cn_stress_profile,
    float *ct_stress_profile, float *l_stress_profile,
    float *fluid_stress_profile, float *vol_frac_profile,
    bool seperate_stresses, float S, float3 director, float S_ref, float3 n_ref,
    std::string& folder_path)
{
    using json = nlohmann::json;
    std::vector<double> vp(velocity_profile,   velocity_profile  +(num_bins+1));
    std::vector<double> gp(gradient_profile,   gradient_profile  +num_bins);
    std::vector<double> sp(total_stress_profile,total_stress_profile+num_bins);
    std::vector<double> cn(cn_stress_profile,  cn_stress_profile +num_bins);
    std::vector<double> ct(ct_stress_profile,  ct_stress_profile +num_bins);
    std::vector<double> ls(l_stress_profile,   l_stress_profile  +num_bins);
    std::vector<double> fs(fluid_stress_profile,fluid_stress_profile+num_bins);
    std::vector<double> vf(vol_frac_profile,   vol_frac_profile  +num_bins);
    double avg_vf = vf.empty() ? 0.0 : std::accumulate(vf.begin(),vf.end(),0.0)/vf.size();
    json j = {
        {"strain",strain},{"time",total_time},{"dt",dt},{"dt_avg",dt_avg},
        {"fluid_viscosity",fluid_viscosity},{"fluid_density",fluid_density},
        {"max_height",max_height},{"velocity_profile",vp},{"gradient_profile",gp},
        {"stress_profile",sp},{"vol_frac_profile",vf},{"overall_vol_frac",avg_vf},
        {"S",S},{"S_ref",S_ref},
        {"director",{director.x,director.y,director.z}},
        {"n_ref",{n_ref.x,n_ref.y,n_ref.z}}
    };
    if (seperate_stresses) {
        j["cont_n_stress_profile"]=cn; j["cont_t_stress_profile"]=ct;
        j["lubr_stress_profile"]=ls; j["fluid_stress_profile"]=fs;
    }
    std::string p = folder_path + std::to_string(frame) + "_fluid_info.json";
    std::ofstream file(p);
    if (file.is_open()) file << j.dump(4);
    else std::cerr << "Error: Could not open file " << p << '\n';
}

void sim_settings_output(float &dt, long long &total_time_steps, float &print_frequency,
    float &save_frequency, float &full_info_frequency, std::string &folder_path)
{
    nlohmann::json j = {{"dt",dt},{"total_time_steps",total_time_steps},
        {"print_frequency",print_frequency},{"save_frequency",save_frequency},
        {"full_info_frequency",full_info_frequency}};
    std::string p = folder_path + "sim_settings.json";
    std::ofstream file(p);
    if (file.is_open()) { file << j.dump(4); file.close(); }
    else std::cerr << "Error: Could not open file " << p << '\n';
}

void system_data_output(float &dt, float3 &system_dimensions, int3 &periodic_boundaries,
    int &contact_toggle, int &friction_toggle, int &lubrication_toggle,
    int &drag_toggle, int &lift_toggle, bool &gravity, float3 &grav_dir,
    std::string &folder_path)
{
    nlohmann::json j = {
        {"dt",dt},
        {"system_dimensions",{system_dimensions.x,system_dimensions.y,system_dimensions.z}},
        {"periodic_boundaries",{periodic_boundaries.x,periodic_boundaries.y,periodic_boundaries.z}},
        {"contact_toggle",contact_toggle},{"friction_toggle",friction_toggle},
        {"lubrication_toggle",lubrication_toggle},{"drag_toggle",drag_toggle},
        {"lift_toggle",lift_toggle},
        {"gravity",gravity},{"grav_dir",{grav_dir.x,grav_dir.y,grav_dir.z}}
    };
    std::string p = folder_path + "system_info.json";
    std::ofstream file(p);
    if (file.is_open()) { file << j.dump(4); file.close(); }
    else std::cerr << "Error: Could not open file " << p << '\n';
}

void createCSVFile(const std::string &folder_path, const std::string &filename,
    const std::vector<std::string> &columns)
{
    std::string filepath = folder_path + filename;
    std::ofstream file(filepath);
    if (file.is_open()) {
        for (size_t i=0;i<columns.size();++i) {
            file << columns[i];
            if (i<columns.size()-1) file << ",";
        }
        file << "\n"; file.close();
    } else std::cerr << "Error: Unable to create file at " << filepath << '\n';
}

void writeCSV(const std::string &folder_path, const std::string &filename,
    const std::vector<double> &rowData, int sig_figs=6)
{
    std::string filepath = folder_path + filename;
    std::ofstream file(filepath, std::ios::app);
    if (file.is_open()) {
        file << std::setprecision(sig_figs);
        for (size_t i=0;i<rowData.size();++i) {
            file << rowData[i];
            if (i<rowData.size()-1) file << ",";
        }
        file << "\n"; file.close();
    } else std::cerr << "Error: Unable to open file for appending: " << filepath << '\n';
}

// ============================================================
//  Order parameters / simulation helpers
// ============================================================

void find_order_parameters(const int num_rods, const float3 *oris, const float3 n_ref,
    float &S, float3 &director, float &S_ref)
{
    Eigen::Matrix3f Q = calc_q_tensor(num_rods, oris);
    S_ref = calc_oop_ref_v1(Q, n_ref);
    auto [Sx, directorx] = calc_oop_and_director(Q);
    S = Sx; director = directorx;
}

void check_print_save_checkpoint(
    int frame, bool use_strains, float tot_strain, float tot_time,
    float &last_print, float &last_save, float &last_checkpoint,
    float print_interval, float save_interval, float checkpoint_interval,
    int timestep, bool &doPrint, bool &doSave, bool &doCheckpoint)
{
    doPrint=false; doSave=false; doCheckpoint=false;
    float ref = use_strains ? tot_strain : tot_time;
    if (ref-last_print >= print_interval) {
        doPrint=true;
        int n=(int)floor((ref-last_print)/print_interval); if(n<1)n=1;
        last_print += n*print_interval;
    }
    if (ref-last_save >= save_interval) {
        doSave=true;
        int n=(int)floor((ref-last_save)/save_interval); if(n<1)n=1;
        last_save += n*save_interval;
    }
    if (ref-last_checkpoint >= checkpoint_interval) {
        doCheckpoint=true;
        int n=(int)floor((ref-last_checkpoint)/checkpoint_interval); if(n<1)n=1;
        last_checkpoint += n*checkpoint_interval;
    }
    if (frame==0) { doPrint=true; doSave=true; doCheckpoint=true; }
}

void print_sequence(bool doPrint, bool use_strain, std::string save_name, float tot_time,
    float tot_strain, float cut_off, float dt, float dt_avg, Timer &clock,
    double speed_wall_s_per_sim_t, Helper helper)
{
    if (doPrint) {
        std::cout << "Simulation: " << save_name << '\n';
        if (use_strain)
            std::cout << "Strain: " << std::fixed << std::setprecision(3) << tot_strain << "/" << cut_off;
        else
            std::cout << "Time: " << std::fixed << std::setprecision(3) << tot_time << "/" << cut_off << " [T]";
        std::cout << "; Time Elapsed: " << clock.time_elapsed();
        if (speed_wall_s_per_sim_t > 0.0)
            std::cout << "; Speed: " << std::fixed << std::setprecision(2)
                      << speed_wall_s_per_sim_t << " wall-s / sim-t";
        std::cout << '\n';
    }
}

void do_LEBCs(float &LEBC_velo, float &LEBC_shift, float *fluid_velocity_profile,
    int num_velos_specified, float dt, float system_x)
{
    LEBC_velo = fluid_velocity_profile[num_velos_specified-1] - fluid_velocity_profile[0];
    LEBC_shift += LEBC_velo * dt;
    LEBC_shift = std::fmod(LEBC_shift, system_x);
    if (LEBC_shift < 0) LEBC_shift += system_x;
}


// ============================================================
//  Params structs (must match MSL structs in rods_kernels.metal exactly)
// ============================================================

struct ClearParams {
    int   num_particles;
    int   gravity;
    float grav_x, grav_y, grav_z;
    int   pad;
};

struct PairParams {
    int   num_particles;
    float dt;
    float viscosity;
    float min_sep;
    float max_sep;
    float ee_manual_weight;
    float ss_manual_weight;
    float es_manual_weight;
    int   contact_toggle;
    int   friction_toggle;
    int   lub_toggle;
    float sys_x, sys_y, sys_z;
    int   pb_x, pb_y, pb_z;
    float LEBC_shift;
    float LEBC_velo;
    int   gen_phase;
    int   pad;
};

struct BodyParams {
    int   num_particles;
    int   contact_toggle;
    int   drag_toggle;
    int   lift_toggle;
    int   num_bins;
    float bin_size;
    float viscosity;
    float fluid_density;
    float max_height;
    float sys_x, sys_y, sys_z;
    int   pb_x, pb_y, pb_z;
    int   gen_phase;
    int   pad;
};

struct IntegrateParams {
    int   num_particles;
    float LEBC_shift;
    float LEBC_velo;
    float dt;
    float sys_x, sys_y, sys_z;
    int   pb_x, pb_y, pb_z;
    int   allowRotation;
};

struct ProfileParams {
    int   num_particles;
    int   num_bins;
    float bin_size;
    float sys_x, sys_y, sys_z;
    int   pb_x, pb_y, pb_z;
    int   pad;
};

struct ScaleParams {
    int   num_particles;
    float scale;
};

struct EnergyParams {
    int num_particles;
};

struct NonAffineParams {
    int   num_particles;
    int   num_grad_bins;
    float bin_size;
    float characteristic_shearrate;
};

// Must mirror PairConsts in rods_kernels.metal exactly.
struct PairConsts {
    float kn_eff;
    float M_eff;
    float en_eff;
    float et_eff;
    float t_c;
    float dc_n;
    float pad0;
    float pad1;
};

// ============================================================
//  MetalSim — device, queue, PSOs, and main simulation buffers
// ============================================================

struct MetalSim {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      lib;

    // Pipeline State Objects for all kernels
    id<MTLComputePipelineState> pso_clear;
    id<MTLComputePipelineState> pso_clear_interactions;
    id<MTLComputePipelineState> pso_pair_interactions;
    id<MTLComputePipelineState> pso_pair_interactions_1D;
    id<MTLComputePipelineState> pso_body_interactions;
    id<MTLComputePipelineState> pso_integrate;
    id<MTLComputePipelineState> pso_get_profiles;
    id<MTLComputePipelineState> pso_scale_CoMs;
    id<MTLComputePipelineState> pso_find_energy;
    id<MTLComputePipelineState> pso_find_nonaffine;
    // Main simulation buffers (MTLResourceStorageModeShared = unified CPU/GPU memory)
    id<MTLBuffer> buf_CoMs;
    id<MTLBuffer> buf_oris;
    id<MTLBuffer> buf_endpoints1;
    id<MTLBuffer> buf_endpoints2;
    id<MTLBuffer> buf_tvels;
    id<MTLBuffer> buf_avels;
    id<MTLBuffer> buf_forces;
    id<MTLBuffer> buf_torques;
    id<MTLBuffer> buf_taccs;
    id<MTLBuffer> buf_aaccs;
    id<MTLBuffer> buf_moi;           // float3 moments of inertia
    id<MTLBuffer> buf_old_int;
    id<MTLBuffer> buf_new_int;
    id<MTLBuffer> buf_shafts;
    id<MTLBuffer> buf_radii;
    id<MTLBuffer> buf_masses;
    id<MTLBuffer> buf_volumes;
    id<MTLBuffer> buf_kns;
    id<MTLBuffer> buf_ens;
    id<MTLBuffer> buf_ets;
    id<MTLBuffer> buf_fric_coefs;
    id<MTLBuffer> buf_avg_inerts;
    id<MTLBuffer> buf_coord_nums;    // int*
    id<MTLBuffer> buf_cn_stresses;   // N*9 floats (mat33 array)
    id<MTLBuffer> buf_ct_stresses;
    id<MTLBuffer> buf_l_stresses;
    id<MTLBuffer> buf_velo_profile;  // float[num_velos_specified]
    id<MTLBuffer> buf_grad_profile;  // float[num_bins]
    id<MTLBuffer> buf_vol_frac_profile;
    id<MTLBuffer> buf_cn_stress_profile;
    id<MTLBuffer> buf_ct_stress_profile;
    id<MTLBuffer> buf_l_stress_profile;
    id<MTLBuffer> buf_min_dt_cont;
    id<MTLBuffer> buf_min_dt_force;
    id<MTLBuffer> buf_min_dt_torque;
    id<MTLBuffer> buf_KE;            // float[N] for find_energy
    id<MTLBuffer> buf_num_particles_ref; // int (for clear_interactions buffer(0))
    id<MTLBuffer> buf_global_max;    // float (for find_if_nonaffine)
    id<MTLBuffer> buf_pair_consts;   // PairConsts[combis]

    int num_particles;
    long long combis;
    int TPB;
    int TPB2;
    int TPB3;
    int num_bins;
    int num_velos_specified;

};

// Helper: allocate a shared buffer
static id<MTLBuffer> newBuf(id<MTLDevice> dev, size_t bytes) {
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

// Helper: create PSO from kernel function name
static id<MTLComputePipelineState> makePSO(id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        NSLog(@"ERROR: Metal function '%@' not found in library", name);
        exit(1);
    }
    id<MTLComputePipelineState> pso = [lib.device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        NSLog(@"ERROR creating PSO for '%@': %@", name, err.localizedDescription);
        exit(1);
    }
    return pso;
}

// Initialise MetalSim: load .metallib, create PSOs
void metal_init(MetalSim &ms, const std::string &metallib_path) {
    ms.device = MTLCreateSystemDefaultDevice();
    if (!ms.device) { std::cerr << "No Metal device found\n"; exit(1); }
    ms.queue = [ms.device newCommandQueue];

    NSString *path = [NSString stringWithUTF8String:metallib_path.c_str()];
    NSError  *err  = nil;
    ms.lib = [ms.device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!ms.lib) {
        NSLog(@"ERROR loading metallib '%@': %@", path, err.localizedDescription);
        exit(1);
    }

    ms.pso_clear                = makePSO(ms.lib, @"clear_for_new_time_step");
    ms.pso_clear_interactions   = makePSO(ms.lib, @"clear_interactions");
    ms.pso_pair_interactions    = makePSO(ms.lib, @"pair_interactions");
    ms.pso_pair_interactions_1D = makePSO(ms.lib, @"pair_interactions_1D");
    ms.pso_body_interactions    = makePSO(ms.lib, @"body_interactions_full");
    ms.pso_integrate          = makePSO(ms.lib, @"integrate_positions");
    ms.pso_get_profiles       = makePSO(ms.lib, @"get_profiles");
    ms.pso_scale_CoMs         = makePSO(ms.lib, @"scale_CoMs");
    ms.pso_find_energy        = makePSO(ms.lib, @"find_energy");
    ms.pso_find_nonaffine     = makePSO(ms.lib, @"find_if_nonaffine_velocity_slow_enough");
}

// Allocate all main simulation buffers
void metal_alloc_buffers(MetalSim &ms, int num_particles, long long combis,
    int num_bins, int num_velos_specified)
{
    ms.num_particles     = num_particles;
    ms.combis            = combis;
    ms.num_bins          = num_bins;
    ms.num_velos_specified = num_velos_specified;

    size_t N = (size_t)num_particles;
    size_t C = (size_t)combis;
    size_t B = (size_t)num_bins;

    ms.buf_CoMs           = newBuf(ms.device, N * sizeof(float3));
    ms.buf_oris           = newBuf(ms.device, N * sizeof(float3));
    ms.buf_endpoints1     = newBuf(ms.device, N * sizeof(float3));
    ms.buf_endpoints2     = newBuf(ms.device, N * sizeof(float3));
    ms.buf_tvels          = newBuf(ms.device, N * sizeof(float3));
    ms.buf_avels          = newBuf(ms.device, N * sizeof(float3));
    ms.buf_forces         = newBuf(ms.device, N * sizeof(float3));
    ms.buf_torques        = newBuf(ms.device, N * sizeof(float3));
    ms.buf_taccs          = newBuf(ms.device, N * sizeof(float3));
    ms.buf_aaccs          = newBuf(ms.device, N * sizeof(float3));
    ms.buf_moi            = newBuf(ms.device, N * sizeof(float3));
    ms.buf_old_int        = newBuf(ms.device, C * sizeof(float3));
    ms.buf_new_int        = newBuf(ms.device, C * sizeof(float3));
    ms.buf_shafts         = newBuf(ms.device, N * sizeof(float));
    ms.buf_radii          = newBuf(ms.device, N * sizeof(float));
    ms.buf_masses         = newBuf(ms.device, N * sizeof(float));
    ms.buf_volumes        = newBuf(ms.device, N * sizeof(float));
    ms.buf_kns            = newBuf(ms.device, N * sizeof(float));
    ms.buf_ens            = newBuf(ms.device, N * sizeof(float));
    ms.buf_ets            = newBuf(ms.device, N * sizeof(float));
    ms.buf_fric_coefs     = newBuf(ms.device, N * sizeof(float));
    ms.buf_avg_inerts     = newBuf(ms.device, N * sizeof(float));
    ms.buf_coord_nums     = newBuf(ms.device, N * sizeof(int));
    ms.buf_cn_stresses    = newBuf(ms.device, N * 9 * sizeof(float));
    ms.buf_ct_stresses    = newBuf(ms.device, N * 9 * sizeof(float));
    ms.buf_l_stresses     = newBuf(ms.device, N * 9 * sizeof(float));
    ms.buf_velo_profile   = newBuf(ms.device, (size_t)num_velos_specified * sizeof(float));
    ms.buf_grad_profile   = newBuf(ms.device, B * sizeof(float));
    ms.buf_vol_frac_profile      = newBuf(ms.device, B * sizeof(float));
    ms.buf_cn_stress_profile     = newBuf(ms.device, B * sizeof(float));
    ms.buf_ct_stress_profile     = newBuf(ms.device, B * sizeof(float));
    ms.buf_l_stress_profile      = newBuf(ms.device, B * sizeof(float));
    ms.buf_min_dt_cont    = newBuf(ms.device, sizeof(float));
    ms.buf_min_dt_force   = newBuf(ms.device, sizeof(float));
    ms.buf_min_dt_torque  = newBuf(ms.device, sizeof(float));
    ms.buf_KE             = newBuf(ms.device, N * sizeof(float));
    ms.buf_num_particles_ref = newBuf(ms.device, sizeof(int));
    ms.buf_global_max     = newBuf(ms.device, sizeof(float));
    ms.buf_pair_consts    = newBuf(ms.device, C * sizeof(PairConsts));
}

// ============================================================
//  Low-level Metal dispatch helpers
// ============================================================

static void commit_wait(id<MTLCommandBuffer> cb) { [cb commit]; [cb waitUntilCompleted]; }

// int_index: must match the MSL version exactly.
static inline int h_int_index(int i, int j, int N) {
    return i * N - (i * (i - 1)) / 2 + (j - 1);
}
// upper_tri_buf_size: required buffer length for the non-compact int_index scheme.
// The max index occurs at (i=N-2, j=N-1) = (N-2)*N - (N-2)*(N-3)/2 + (N-2) = (N-2)*(N+5)/2.
static inline long long upper_tri_buf_size(int N) {
    if (N <= 1) return 1LL;
    return (long long)(N-2) * (long long)(N+5) / 2LL + 1LL;
}

// Precompute constant per-pair contact parameters into buf_pair_consts.
// Call once after particle data is written to GPU buffers.
static void precompute_pair_consts(MetalSim &ms,
    const float* kns, const float* ens, const float* ets,
    const float* masses, int num_particles)
{
    PairConsts* pc = (PairConsts*)ms.buf_pair_consts.contents;
    for (int i = 0; i < num_particles; ++i) {
        for (int j = i + 1; j < num_particles; ++j) {
            int idx     = h_int_index(i, j, num_particles);
            float kn_e  = kns[i] * kns[j] / (kns[i] + kns[j]);
            float M_e   = masses[i] * masses[j] / (masses[i] + masses[j]);
            float en_e  = std::fmin(ens[i], ens[j]);
            float et_e  = std::fmin(ets[i], ets[j]);
            float log_en = std::log(en_e);
            float t_c    = std::sqrt(h_PI*h_PI + log_en*log_en) * std::sqrt(M_e / kn_e);
            float dc_n   = (std::isfinite(t_c) && t_c > 0.0f)
                           ? -2.0f * M_e / t_c * log_en : 0.0f;
            pc[idx].kn_eff = kn_e;
            pc[idx].M_eff  = M_e;
            pc[idx].en_eff = en_e;
            pc[idx].et_eff = et_e;
            pc[idx].t_c    = t_c;
            pc[idx].dc_n   = dc_n;
            pc[idx].pad0   = 0.0f;
            pc[idx].pad1   = 0.0f;
        }
    }
}

// dispatch_clear_for_new_time_step
// All buffer args are explicit so relaxation() can pass its own local buffers
static void encode_clear_kernel(
    id<MTLCommandBuffer> cb,
    MetalSim &ms, int num_particles, bool gravity, float3 grav_dir,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_cn, id<MTLBuffer> b_ct, id<MTLBuffer> b_l,
    id<MTLBuffer> b_masses, id<MTLBuffer> b_coord_nums)
{
    ClearParams p{};
    p.num_particles = num_particles;
    p.gravity = (int)gravity;
    p.grav_x = grav_dir.x; p.grav_y = grav_dir.y; p.grav_z = grav_dir.z;
    p.pad = 0;

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ms.pso_clear];
    [enc setBytes:&p length:sizeof(p) atIndex:0];
    [enc setBuffer:b_forces   offset:0 atIndex:1];
    [enc setBuffer:b_torques  offset:0 atIndex:2];
    [enc setBuffer:b_cn       offset:0 atIndex:3];
    [enc setBuffer:b_ct       offset:0 atIndex:4];
    [enc setBuffer:b_l        offset:0 atIndex:5];
    [enc setBuffer:b_masses   offset:0 atIndex:6];
    [enc setBuffer:b_coord_nums offset:0 atIndex:7];
    NSUInteger tgSize = (NSUInteger)ms.TPB;
    NSUInteger gridSz = (NSUInteger)num_particles;
    [enc dispatchThreads:MTLSizeMake(gridSz,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
    [enc endEncoding];
}

static void dispatch_clear_kernel(
    MetalSim &ms, int num_particles, bool gravity, float3 grav_dir,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_cn, id<MTLBuffer> b_ct, id<MTLBuffer> b_l,
    id<MTLBuffer> b_masses, id<MTLBuffer> b_coord_nums)
{
    id<MTLCommandBuffer> cb = [ms.queue commandBuffer];
    encode_clear_kernel(cb, ms, num_particles, gravity, grav_dir,
        b_forces, b_torques, b_cn, b_ct, b_l, b_masses, b_coord_nums);
    commit_wait(cb);
}

// dispatch_clear_interactions
static void encode_clear_interactions_kernel(
    id<MTLCommandBuffer> cb,
    MetalSim &ms, int num_particles, long long combis,
    id<MTLBuffer> b_num_ref, id<MTLBuffer> b_old, id<MTLBuffer> b_new)
{
    *(int*)b_num_ref.contents = num_particles;  // CPU write — must happen before [cb commit]
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ms.pso_clear_interactions];
    [enc setBuffer:b_num_ref offset:0 atIndex:0];
    [enc setBuffer:b_old     offset:0 atIndex:1];
    [enc setBuffer:b_new     offset:0 atIndex:2];
    NSUInteger gridSz = (NSUInteger)combis;
    NSUInteger tgSize = (NSUInteger)ms.TPB3;
    [enc dispatchThreads:MTLSizeMake(gridSz,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
    [enc endEncoding];
}

static void dispatch_clear_interactions_kernel(
    MetalSim &ms, int num_particles, long long combis,
    id<MTLBuffer> b_num_ref, id<MTLBuffer> b_old, id<MTLBuffer> b_new)
{
    id<MTLCommandBuffer> cb = [ms.queue commandBuffer];
    encode_clear_interactions_kernel(cb, ms, num_particles, combis, b_num_ref, b_old, b_new);
    commit_wait(cb);
}

// dispatch_pair_interactions
static void encode_pair_kernel(
    id<MTLCommandBuffer> cb,
    MetalSim &ms, int num_particles,
    float dt, float viscosity, float min_sep, float max_sep,
    float ee_w, float ss_w, float es_w,
    int contact_toggle, int friction_toggle, int lub_toggle,
    float3 sys_dim, int3 pb, float LEBC_shift, float LEBC_velo, bool gen_phase,
    id<MTLBuffer> b_CoMs, id<MTLBuffer> b_oris, id<MTLBuffer> b_shafts,
    id<MTLBuffer> b_radii, id<MTLBuffer> b_masses, id<MTLBuffer> b_avg_inerts,
    id<MTLBuffer> b_coord_nums,
    id<MTLBuffer> b_fric,
    id<MTLBuffer> b_tvels, id<MTLBuffer> b_avels,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_cn, id<MTLBuffer> b_ct, id<MTLBuffer> b_l,
    id<MTLBuffer> b_old_int, id<MTLBuffer> b_new_int,
    id<MTLBuffer> b_moi,
    id<MTLBuffer> b_min_dt_cont, id<MTLBuffer> b_min_dt_force, id<MTLBuffer> b_min_dt_torque,
    id<MTLBuffer> b_pair_consts)
{
    PairParams p{};
    p.num_particles = num_particles; p.dt = dt; p.viscosity = viscosity;
    p.min_sep = min_sep; p.max_sep = max_sep;
    p.ee_manual_weight = ee_w; p.ss_manual_weight = ss_w; p.es_manual_weight = es_w;
    p.contact_toggle = contact_toggle; p.friction_toggle = friction_toggle; p.lub_toggle = lub_toggle;
    p.sys_x = sys_dim.x; p.sys_y = sys_dim.y; p.sys_z = sys_dim.z;
    p.pb_x = pb.x; p.pb_y = pb.y; p.pb_z = pb.z;
    p.LEBC_shift = LEBC_shift; p.LEBC_velo = LEBC_velo;
    p.gen_phase = (int)gen_phase; p.pad = 0;

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ms.pso_pair_interactions];
    [enc setBytes:&p length:sizeof(p) atIndex:0];
    [enc setBuffer:b_CoMs       offset:0 atIndex:1];
    [enc setBuffer:b_oris       offset:0 atIndex:2];
    [enc setBuffer:b_shafts     offset:0 atIndex:3];
    [enc setBuffer:b_radii      offset:0 atIndex:4];
    [enc setBuffer:b_masses     offset:0 atIndex:5];
    [enc setBuffer:b_avg_inerts offset:0 atIndex:6];
    [enc setBuffer:b_coord_nums offset:0 atIndex:7];
    // slots 8,9,10 (kns,ens,ets) unused — values come from pair_consts at slot 25
    [enc setBuffer:b_fric       offset:0 atIndex:11];
    [enc setBuffer:b_tvels      offset:0 atIndex:12];
    [enc setBuffer:b_avels      offset:0 atIndex:13];
    [enc setBuffer:b_forces     offset:0 atIndex:14];
    [enc setBuffer:b_torques    offset:0 atIndex:15];
    [enc setBuffer:b_cn         offset:0 atIndex:16];
    [enc setBuffer:b_ct         offset:0 atIndex:17];
    [enc setBuffer:b_l          offset:0 atIndex:18];
    [enc setBuffer:b_old_int    offset:0 atIndex:19];
    [enc setBuffer:b_new_int    offset:0 atIndex:20];
    [enc setBuffer:b_moi        offset:0 atIndex:21];
    [enc setBuffer:b_min_dt_cont   offset:0 atIndex:22];
    [enc setBuffer:b_min_dt_force  offset:0 atIndex:23];
    [enc setBuffer:b_min_dt_torque offset:0 atIndex:24];
    [enc setBuffer:b_pair_consts   offset:0 atIndex:25];
    NSUInteger tg = (NSUInteger)ms.TPB2;
    NSUInteger N  = (NSUInteger)num_particles;
    [enc dispatchThreads:MTLSizeMake(N,N,1) threadsPerThreadgroup:MTLSizeMake(tg,tg,1)];
    [enc endEncoding];
}

// encode_pair_kernel_1D — 1D triangular dispatch using pair_interactions_1D kernel.
static void encode_pair_kernel_1D(
    id<MTLCommandBuffer> cb,
    MetalSim &ms, int num_particles,
    float dt, float viscosity, float min_sep, float max_sep,
    float ee_w, float ss_w, float es_w,
    int contact_toggle, int friction_toggle, int lub_toggle,
    float3 sys_dim, int3 pb, float LEBC_shift, float LEBC_velo, bool gen_phase,
    id<MTLBuffer> b_CoMs, id<MTLBuffer> b_oris, id<MTLBuffer> b_shafts,
    id<MTLBuffer> b_radii, id<MTLBuffer> b_masses, id<MTLBuffer> b_avg_inerts,
    id<MTLBuffer> b_coord_nums,
    id<MTLBuffer> b_fric,
    id<MTLBuffer> b_tvels, id<MTLBuffer> b_avels,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_cn, id<MTLBuffer> b_ct, id<MTLBuffer> b_l,
    id<MTLBuffer> b_old_int, id<MTLBuffer> b_new_int,
    id<MTLBuffer> b_moi,
    id<MTLBuffer> b_min_dt_cont, id<MTLBuffer> b_min_dt_force, id<MTLBuffer> b_min_dt_torque,
    id<MTLBuffer> b_pair_consts)
{
    PairParams p{};
    p.num_particles = num_particles; p.dt = dt; p.viscosity = viscosity;
    p.min_sep = min_sep; p.max_sep = max_sep;
    p.ee_manual_weight = ee_w; p.ss_manual_weight = ss_w; p.es_manual_weight = es_w;
    p.contact_toggle = contact_toggle; p.friction_toggle = friction_toggle; p.lub_toggle = lub_toggle;
    p.sys_x = sys_dim.x; p.sys_y = sys_dim.y; p.sys_z = sys_dim.z;
    p.pb_x = pb.x; p.pb_y = pb.y; p.pb_z = pb.z;
    p.LEBC_shift = LEBC_shift; p.LEBC_velo = LEBC_velo;
    p.gen_phase = (int)gen_phase; p.pad = 0;

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ms.pso_pair_interactions_1D];
    [enc setBytes:&p          length:sizeof(p) atIndex:0];
    [enc setBuffer:b_CoMs       offset:0 atIndex:1];
    [enc setBuffer:b_oris       offset:0 atIndex:2];
    [enc setBuffer:b_shafts     offset:0 atIndex:3];
    [enc setBuffer:b_radii      offset:0 atIndex:4];
    [enc setBuffer:b_masses     offset:0 atIndex:5];
    [enc setBuffer:b_avg_inerts offset:0 atIndex:6];
    [enc setBuffer:b_coord_nums offset:0 atIndex:7];
    // slots 8,9,10 (kns,ens,ets) unused in 1D kernel — pair_consts at 25 replaces them
    [enc setBuffer:b_fric       offset:0 atIndex:11];
    [enc setBuffer:b_tvels      offset:0 atIndex:12];
    [enc setBuffer:b_avels      offset:0 atIndex:13];
    [enc setBuffer:b_forces     offset:0 atIndex:14];
    [enc setBuffer:b_torques    offset:0 atIndex:15];
    [enc setBuffer:b_cn         offset:0 atIndex:16];
    [enc setBuffer:b_ct         offset:0 atIndex:17];
    [enc setBuffer:b_l          offset:0 atIndex:18];
    [enc setBuffer:b_old_int    offset:0 atIndex:19];
    [enc setBuffer:b_new_int    offset:0 atIndex:20];
    [enc setBuffer:b_moi        offset:0 atIndex:21];
    [enc setBuffer:b_min_dt_cont   offset:0 atIndex:22];
    [enc setBuffer:b_min_dt_force  offset:0 atIndex:23];
    [enc setBuffer:b_min_dt_torque offset:0 atIndex:24];
    [enc setBuffer:b_pair_consts   offset:0 atIndex:25];
    long long combis_1D = (long long)num_particles * ((long long)num_particles - 1LL) / 2LL;
    NSUInteger gridSz = (NSUInteger)combis_1D;
    NSUInteger tgSize = (NSUInteger)ms.TPB3;
    [enc dispatchThreads:MTLSizeMake(gridSz,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
    [enc endEncoding];
}

static void dispatch_pair_kernel(
    MetalSim &ms, int num_particles,
    float dt, float viscosity, float min_sep, float max_sep,
    float ee_w, float ss_w, float es_w,
    int contact_toggle, int friction_toggle, int lub_toggle,
    float3 sys_dim, int3 pb, float LEBC_shift, float LEBC_velo, bool gen_phase,
    id<MTLBuffer> b_CoMs, id<MTLBuffer> b_oris, id<MTLBuffer> b_shafts,
    id<MTLBuffer> b_radii, id<MTLBuffer> b_masses, id<MTLBuffer> b_avg_inerts,
    id<MTLBuffer> b_coord_nums, id<MTLBuffer> b_fric,
    id<MTLBuffer> b_tvels, id<MTLBuffer> b_avels,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_cn, id<MTLBuffer> b_ct, id<MTLBuffer> b_l,
    id<MTLBuffer> b_old_int, id<MTLBuffer> b_new_int, id<MTLBuffer> b_moi,
    id<MTLBuffer> b_min_dt_cont, id<MTLBuffer> b_min_dt_force, id<MTLBuffer> b_min_dt_torque,
    id<MTLBuffer> b_pair_consts)
{
    id<MTLCommandBuffer> cb = [ms.queue commandBuffer];
    encode_pair_kernel(cb, ms, num_particles, dt, viscosity, min_sep, max_sep,
        ee_w, ss_w, es_w, contact_toggle, friction_toggle, lub_toggle,
        sys_dim, pb, LEBC_shift, LEBC_velo, gen_phase,
        b_CoMs, b_oris, b_shafts, b_radii, b_masses, b_avg_inerts, b_coord_nums,
        b_fric, b_tvels, b_avels,
        b_forces, b_torques, b_cn, b_ct, b_l,
        b_old_int, b_new_int, b_moi,
        b_min_dt_cont, b_min_dt_force, b_min_dt_torque, b_pair_consts);
    commit_wait(cb);
}

// dispatch_body_interactions
static void encode_body_kernel(
    id<MTLCommandBuffer> cb,
    MetalSim &ms, int num_particles, int3 pb, float3 sys_dim,
    int contact_toggle, int drag_toggle, int lift_toggle,
    int num_bins, float bin_size, float viscosity, float fluid_density, float max_height,
    bool gen_phase,
    id<MTLBuffer> b_CoMs, id<MTLBuffer> b_oris,
    id<MTLBuffer> b_ep1, id<MTLBuffer> b_ep2,
    id<MTLBuffer> b_radii, id<MTLBuffer> b_shafts, id<MTLBuffer> b_masses,
    id<MTLBuffer> b_kns, id<MTLBuffer> b_ens,
    id<MTLBuffer> b_tvels, id<MTLBuffer> b_avels,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_velo_profile, id<MTLBuffer> b_grad_profile,
    id<MTLBuffer> b_coord_nums)
{
    BodyParams p{};
    p.num_particles = num_particles; p.contact_toggle = contact_toggle;
    p.drag_toggle = drag_toggle; p.lift_toggle = lift_toggle;
    p.num_bins = num_bins; p.bin_size = bin_size;
    p.viscosity = viscosity; p.fluid_density = fluid_density; p.max_height = max_height;
    p.sys_x = sys_dim.x; p.sys_y = sys_dim.y; p.sys_z = sys_dim.z;
    p.pb_x = pb.x; p.pb_y = pb.y; p.pb_z = pb.z;
    p.gen_phase = (int)gen_phase; p.pad = 0;

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ms.pso_body_interactions];
    [enc setBytes:&p length:sizeof(p) atIndex:0];
    [enc setBuffer:b_CoMs    offset:0 atIndex:1];
    [enc setBuffer:b_oris    offset:0 atIndex:2];
    [enc setBuffer:b_ep1     offset:0 atIndex:3];
    [enc setBuffer:b_ep2     offset:0 atIndex:4];
    [enc setBuffer:b_radii   offset:0 atIndex:5];
    [enc setBuffer:b_shafts  offset:0 atIndex:6];
    [enc setBuffer:b_masses  offset:0 atIndex:7];
    [enc setBuffer:b_kns     offset:0 atIndex:8];
    [enc setBuffer:b_ens     offset:0 atIndex:9];
    [enc setBuffer:b_tvels   offset:0 atIndex:10];
    [enc setBuffer:b_avels   offset:0 atIndex:11];
    [enc setBuffer:b_forces  offset:0 atIndex:12];
    [enc setBuffer:b_torques offset:0 atIndex:13];
    [enc setBuffer:b_velo_profile offset:0 atIndex:14];
    [enc setBuffer:b_grad_profile offset:0 atIndex:15];
    [enc setBuffer:b_coord_nums   offset:0 atIndex:16];
    NSUInteger gridSz = (NSUInteger)num_particles;
    NSUInteger tgSize = (NSUInteger)ms.TPB;
    [enc dispatchThreads:MTLSizeMake(gridSz,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
    [enc endEncoding];
}

static void dispatch_body_kernel(
    MetalSim &ms, int num_particles, int3 pb, float3 sys_dim,
    int contact_toggle, int drag_toggle, int lift_toggle,
    int num_bins, float bin_size, float viscosity, float fluid_density, float max_height,
    bool gen_phase,
    id<MTLBuffer> b_CoMs, id<MTLBuffer> b_oris,
    id<MTLBuffer> b_ep1, id<MTLBuffer> b_ep2,
    id<MTLBuffer> b_radii, id<MTLBuffer> b_shafts, id<MTLBuffer> b_masses,
    id<MTLBuffer> b_kns, id<MTLBuffer> b_ens,
    id<MTLBuffer> b_tvels, id<MTLBuffer> b_avels,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_velo_profile, id<MTLBuffer> b_grad_profile,
    id<MTLBuffer> b_coord_nums)
{
    id<MTLCommandBuffer> cb = [ms.queue commandBuffer];
    encode_body_kernel(cb, ms, num_particles, pb, sys_dim,
        contact_toggle, drag_toggle, lift_toggle,
        num_bins, bin_size, viscosity, fluid_density, max_height, gen_phase,
        b_CoMs, b_oris, b_ep1, b_ep2, b_radii, b_shafts, b_masses,
        b_kns, b_ens, b_tvels, b_avels, b_forces, b_torques,
        b_velo_profile, b_grad_profile, b_coord_nums);
    commit_wait(cb);
}

// dispatch_integrate_positions
static void encode_integrate_kernel(
    id<MTLCommandBuffer> cb,
    MetalSim &ms, int num_particles, float3 sys_dim, int3 pb,
    float LEBC_shift, float LEBC_velo, float dt, bool allowRotation,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_CoMs, id<MTLBuffer> b_tvels,
    id<MTLBuffer> b_taccs, id<MTLBuffer> b_masses, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_oris, id<MTLBuffer> b_ep1, id<MTLBuffer> b_ep2,
    id<MTLBuffer> b_shafts, id<MTLBuffer> b_avels, id<MTLBuffer> b_aaccs, id<MTLBuffer> b_moi)
{
    IntegrateParams p{};
    p.num_particles = num_particles;
    p.LEBC_shift = LEBC_shift; p.LEBC_velo = LEBC_velo; p.dt = dt;
    p.sys_x = sys_dim.x; p.sys_y = sys_dim.y; p.sys_z = sys_dim.z;
    p.pb_x = pb.x; p.pb_y = pb.y; p.pb_z = pb.z;
    p.allowRotation = (int)allowRotation;

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ms.pso_integrate];
    [enc setBytes:&p length:sizeof(p) atIndex:0];
    [enc setBuffer:b_forces  offset:0 atIndex:1];
    [enc setBuffer:b_CoMs    offset:0 atIndex:2];
    [enc setBuffer:b_tvels   offset:0 atIndex:3];
    [enc setBuffer:b_taccs   offset:0 atIndex:4];
    [enc setBuffer:b_masses  offset:0 atIndex:5];
    [enc setBuffer:b_torques offset:0 atIndex:6];
    [enc setBuffer:b_oris    offset:0 atIndex:7];
    [enc setBuffer:b_ep1     offset:0 atIndex:8];
    [enc setBuffer:b_ep2     offset:0 atIndex:9];
    [enc setBuffer:b_shafts  offset:0 atIndex:10];
    [enc setBuffer:b_avels   offset:0 atIndex:11];
    [enc setBuffer:b_aaccs   offset:0 atIndex:12];
    [enc setBuffer:b_moi     offset:0 atIndex:13];
    NSUInteger gridSz = (NSUInteger)num_particles;
    NSUInteger tgSize = (NSUInteger)ms.TPB;
    [enc dispatchThreads:MTLSizeMake(gridSz,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
    [enc endEncoding];
}

static void dispatch_integrate_kernel(
    MetalSim &ms, int num_particles, float3 sys_dim, int3 pb,
    float LEBC_shift, float LEBC_velo, float dt, bool allowRotation,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_CoMs, id<MTLBuffer> b_tvels,
    id<MTLBuffer> b_taccs, id<MTLBuffer> b_masses, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_oris, id<MTLBuffer> b_ep1, id<MTLBuffer> b_ep2,
    id<MTLBuffer> b_shafts, id<MTLBuffer> b_avels, id<MTLBuffer> b_aaccs, id<MTLBuffer> b_moi)
{
    id<MTLCommandBuffer> cb = [ms.queue commandBuffer];
    encode_integrate_kernel(cb, ms, num_particles, sys_dim, pb,
        LEBC_shift, LEBC_velo, dt, allowRotation,
        b_forces, b_CoMs, b_tvels, b_taccs, b_masses, b_torques,
        b_oris, b_ep1, b_ep2, b_shafts, b_avels, b_aaccs, b_moi);
    commit_wait(cb);
}

// ─────────────────────────────────────────────────────────
//  BATCHED STEP FUNCTIONS
//  Encode clear + clear_interactions + pair + body + integrate into ONE
//  MTLCommandBuffer per timestep, eliminating 4 out of 5 commit_wait calls.
//  Metal guarantees compute encoders within a command buffer execute in order.
// ─────────────────────────────────────────────────────────

// forward declaration (step_gpu is defined after step_gpu_multi)
static void step_gpu(MetalSim &ms, bool gravity, float3 grav_dir, bool do_friction,
    float dt, float viscosity, float min_sep, float max_sep,
    float ee_w, float ss_w, float es_w,
    int contact_toggle, int friction_toggle, int lub_toggle,
    float3 sys_dim, int3 pb, float LEBC_shift, float LEBC_velo,
    int drag_toggle, int lift_toggle,
    int num_bins, float bin_size, float fluid_density, float max_height);

// step_gpu_multi — runs M simulation steps.
// The cell list requires a CPU prefix sum between GPU passes, so we call
// step_gpu M times.  The per-step overhead is small relative to the speedup
// from only checking nearby pairs instead of all N(N-1)/2 pairs.
static void step_gpu_multi(
    MetalSim &ms, int M, const float* lebc_shifts,
    bool gravity, float3 grav_dir, bool do_friction,
    float dt, float viscosity, float min_sep, float max_sep,
    float ee_w, float ss_w, float es_w,
    int contact_toggle, int friction_toggle, int lub_toggle,
    float3 sys_dim, int3 pb, float LEBC_velo,
    int drag_toggle, int lift_toggle,
    int num_bins, float bin_size, float fluid_density, float max_height)
{
    for (int k = 0; k < M; ++k) {
        step_gpu(ms, gravity, grav_dir, do_friction,
            dt, viscosity, min_sep, max_sep, ee_w, ss_w, es_w,
            contact_toggle, friction_toggle, lub_toggle,
            sys_dim, pb, lebc_shifts[k], LEBC_velo,
            drag_toggle, lift_toggle,
            num_bins, bin_size, fluid_density, max_height);
    }
}

// step_gpu — uses ms.buf_* directly; for the main simulation loop.
// Main simulation step — uses pair_interactions_1D (N*(N-1)/2 threads) for
// maximum GPU occupancy at the particle counts used here.
static void step_gpu(
    MetalSim &ms,
    bool gravity, float3 grav_dir, bool do_friction,
    float dt, float viscosity, float min_sep, float max_sep,
    float ee_w, float ss_w, float es_w,
    int contact_toggle, int friction_toggle, int lub_toggle,
    float3 sys_dim, int3 pb, float LEBC_shift, float LEBC_velo,
    int drag_toggle, int lift_toggle,
    int num_bins, float bin_size, float fluid_density, float max_height)
{
    id<MTLCommandBuffer> cb = [ms.queue commandBuffer];

    encode_clear_kernel(cb, ms, ms.num_particles, gravity, grav_dir,
        ms.buf_forces, ms.buf_torques,
        ms.buf_cn_stresses, ms.buf_ct_stresses, ms.buf_l_stresses,
        ms.buf_masses, ms.buf_coord_nums);

    encode_pair_kernel_1D(cb, ms, ms.num_particles,
        dt, viscosity, min_sep, max_sep, ee_w, ss_w, es_w,
        contact_toggle, friction_toggle, lub_toggle,
        sys_dim, pb, LEBC_shift, LEBC_velo, /*gen_phase=*/false,
        ms.buf_CoMs, ms.buf_oris, ms.buf_shafts,
        ms.buf_radii, ms.buf_masses, ms.buf_avg_inerts, ms.buf_coord_nums,
        ms.buf_fric_coefs,
        ms.buf_tvels, ms.buf_avels,
        ms.buf_forces, ms.buf_torques,
        ms.buf_cn_stresses, ms.buf_ct_stresses, ms.buf_l_stresses,
        ms.buf_old_int, ms.buf_new_int, ms.buf_moi,
        ms.buf_min_dt_cont, ms.buf_min_dt_force, ms.buf_min_dt_torque,
        ms.buf_pair_consts);

    encode_body_kernel(cb, ms, ms.num_particles, pb, sys_dim,
        contact_toggle, drag_toggle, lift_toggle,
        num_bins, bin_size, viscosity, fluid_density, max_height, /*gen_phase=*/false,
        ms.buf_CoMs, ms.buf_oris, ms.buf_endpoints1, ms.buf_endpoints2,
        ms.buf_radii, ms.buf_shafts, ms.buf_masses,
        ms.buf_kns, ms.buf_ens,
        ms.buf_tvels, ms.buf_avels, ms.buf_forces, ms.buf_torques,
        ms.buf_velo_profile, ms.buf_grad_profile, ms.buf_coord_nums);

    encode_integrate_kernel(cb, ms, ms.num_particles, sys_dim, pb,
        LEBC_shift, LEBC_velo, dt, /*allowRotation=*/true,
        ms.buf_forces, ms.buf_CoMs, ms.buf_tvels, ms.buf_taccs, ms.buf_masses,
        ms.buf_torques, ms.buf_oris, ms.buf_endpoints1, ms.buf_endpoints2,
        ms.buf_shafts, ms.buf_avels, ms.buf_aaccs, ms.buf_moi);

    commit_wait(cb);
}

// step_gpu_relax — takes explicit buffers; for the relaxation loop.
// Uses pair_interactions_1D for maximum GPU occupancy.
static void step_gpu_relax(
    MetalSim &ms, int num_particles, long long combis,
    bool gravity, float3 grav_dir, bool do_friction,
    float dt, float viscosity, float min_sep, float max_sep,
    float ee_w, float ss_w, float es_w,
    int contact_toggle, int friction_toggle, int lub_toggle,
    float3 sys_dim, int3 pb, float LEBC_shift, float LEBC_velo, bool gen_phase,
    int drag_toggle, int lift_toggle,
    int num_bins, float bin_size, float fluid_density, float max_height,
    bool allowRotation,
    id<MTLBuffer> b_CoMs,    id<MTLBuffer> b_oris,
    id<MTLBuffer> b_ep1,     id<MTLBuffer> b_ep2,
    id<MTLBuffer> b_tvels,   id<MTLBuffer> b_avels,
    id<MTLBuffer> b_forces,  id<MTLBuffer> b_torques,
    id<MTLBuffer> b_taccs,   id<MTLBuffer> b_aaccs,
    id<MTLBuffer> b_masses,  id<MTLBuffer> b_shafts,  id<MTLBuffer> b_radii,
    id<MTLBuffer> b_kns,     id<MTLBuffer> b_ens,
    id<MTLBuffer> b_ets,     id<MTLBuffer> b_fric,
    id<MTLBuffer> b_avg_inerts, id<MTLBuffer> b_coord_nums,
    id<MTLBuffer> b_cn,      id<MTLBuffer> b_ct,      id<MTLBuffer> b_l,
    id<MTLBuffer> b_old_int, id<MTLBuffer> b_new_int, id<MTLBuffer> b_moi,
    id<MTLBuffer> b_num_ref,
    id<MTLBuffer> b_velo,    id<MTLBuffer> b_grad,
    id<MTLBuffer> b_min_dt_cont, id<MTLBuffer> b_min_dt_force, id<MTLBuffer> b_min_dt_torque)
{
    id<MTLCommandBuffer> cb = [ms.queue commandBuffer];

    encode_clear_kernel(cb, ms, num_particles, gravity, grav_dir,
        b_forces, b_torques, b_cn, b_ct, b_l, b_masses, b_coord_nums);

    encode_pair_kernel_1D(cb, ms, num_particles,
        dt, viscosity, min_sep, max_sep, ee_w, ss_w, es_w,
        contact_toggle, friction_toggle, lub_toggle,
        sys_dim, pb, LEBC_shift, LEBC_velo, gen_phase,
        b_CoMs, b_oris, b_shafts, b_radii, b_masses, b_avg_inerts, b_coord_nums,
        b_fric, b_tvels, b_avels,
        b_forces, b_torques, b_cn, b_ct, b_l,
        b_old_int, b_new_int, b_moi,
        b_min_dt_cont, b_min_dt_force, b_min_dt_torque, ms.buf_pair_consts);

    encode_body_kernel(cb, ms, num_particles, pb, sys_dim,
        contact_toggle, drag_toggle, lift_toggle,
        num_bins, bin_size, viscosity, fluid_density, max_height, gen_phase,
        b_CoMs, b_oris, b_ep1, b_ep2, b_radii, b_shafts, b_masses,
        b_kns, b_ens, b_tvels, b_avels, b_forces, b_torques,
        b_velo, b_grad, b_coord_nums);

    encode_integrate_kernel(cb, ms, num_particles, sys_dim, pb,
        LEBC_shift, LEBC_velo, dt, allowRotation,
        b_forces, b_CoMs, b_tvels, b_taccs, b_masses, b_torques,
        b_oris, b_ep1, b_ep2, b_shafts, b_avels, b_aaccs, b_moi);

    commit_wait(cb);
}

// dispatch_get_profiles
static void dispatch_get_profiles_kernel(
    MetalSim &ms, int num_particles, int num_bins, float bin_size,
    float3 sys_dim, int3 pb,
    id<MTLBuffer> b_CoMs, id<MTLBuffer> b_ep1, id<MTLBuffer> b_ep2,
    id<MTLBuffer> b_volumes, id<MTLBuffer> b_vol_frac,
    id<MTLBuffer> b_cn, id<MTLBuffer> b_ct, id<MTLBuffer> b_l,
    id<MTLBuffer> b_cn_prof, id<MTLBuffer> b_ct_prof, id<MTLBuffer> b_l_prof)
{
    ProfileParams p{};
    p.num_particles = num_particles; p.num_bins = num_bins; p.bin_size = bin_size;
    p.sys_x = sys_dim.x; p.sys_y = sys_dim.y; p.sys_z = sys_dim.z;
    p.pb_x = pb.x; p.pb_y = pb.y; p.pb_z = pb.z; p.pad = 0;

    id<MTLCommandBuffer> cb = [ms.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ms.pso_get_profiles];
    [enc setBytes:&p length:sizeof(p) atIndex:0];
    [enc setBuffer:b_CoMs    offset:0 atIndex:1];
    [enc setBuffer:b_ep1     offset:0 atIndex:2];
    [enc setBuffer:b_ep2     offset:0 atIndex:3];
    [enc setBuffer:b_volumes offset:0 atIndex:4];
    [enc setBuffer:b_vol_frac offset:0 atIndex:5];
    [enc setBuffer:b_cn      offset:0 atIndex:6];
    [enc setBuffer:b_ct      offset:0 atIndex:7];
    [enc setBuffer:b_l       offset:0 atIndex:8];
    [enc setBuffer:b_cn_prof offset:0 atIndex:9];
    [enc setBuffer:b_ct_prof offset:0 atIndex:10];
    [enc setBuffer:b_l_prof  offset:0 atIndex:11];
    NSUInteger gridSz = (NSUInteger)num_particles;
    NSUInteger tgSize = (NSUInteger)ms.TPB;
    [enc dispatchThreads:MTLSizeMake(gridSz,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
    [enc endEncoding];
    commit_wait(cb);
}

// dispatch_scale_CoMs
static void dispatch_scale_kernel(MetalSim &ms, int num_particles, float scale,
    id<MTLBuffer> b_CoMs)
{
    ScaleParams p{}; p.num_particles = num_particles; p.scale = scale;
    id<MTLCommandBuffer> cb = [ms.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ms.pso_scale_CoMs];
    [enc setBytes:&p length:sizeof(p) atIndex:0];
    [enc setBuffer:b_CoMs offset:0 atIndex:1];
    NSUInteger gridSz = (NSUInteger)num_particles;
    NSUInteger tgSize = (NSUInteger)ms.TPB;
    [enc dispatchThreads:MTLSizeMake(gridSz,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
    [enc endEncoding];
    commit_wait(cb);
}

// dispatch_find_energy
static void dispatch_find_energy_kernel(MetalSim &ms, int num_particles,
    id<MTLBuffer> b_masses, id<MTLBuffer> b_tvels,
    id<MTLBuffer> b_moi,    id<MTLBuffer> b_avels, id<MTLBuffer> b_KE)
{
    EnergyParams p{}; p.num_particles = num_particles;
    id<MTLCommandBuffer> cb = [ms.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ms.pso_find_energy];
    [enc setBytes:&p length:sizeof(p) atIndex:0];
    [enc setBuffer:b_masses offset:0 atIndex:1];
    [enc setBuffer:b_tvels  offset:0 atIndex:2];
    [enc setBuffer:b_moi    offset:0 atIndex:3];
    [enc setBuffer:b_avels  offset:0 atIndex:4];
    [enc setBuffer:b_KE     offset:0 atIndex:5];
    NSUInteger gridSz = (NSUInteger)num_particles;
    NSUInteger tgSize = (NSUInteger)ms.TPB;
    [enc dispatchThreads:MTLSizeMake(gridSz,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
    [enc endEncoding];
    commit_wait(cb);
}

// dispatch_find_if_nonaffine
static void dispatch_nonaffine_kernel(MetalSim &ms, int num_particles, int num_grad_bins,
    float bin_size, float characteristic_shearrate,
    id<MTLBuffer> b_tvels, id<MTLBuffer> b_CoMs,
    id<MTLBuffer> b_velo_prof, id<MTLBuffer> b_grad_prof,
    id<MTLBuffer> b_global_max, id<MTLBuffer> b_radii)
{
    NonAffineParams p{};
    p.num_particles = num_particles; p.num_grad_bins = num_grad_bins;
    p.bin_size = bin_size; p.characteristic_shearrate = characteristic_shearrate;
    id<MTLCommandBuffer> cb = [ms.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ms.pso_find_nonaffine];
    [enc setBytes:&p length:sizeof(p) atIndex:0];
    [enc setBuffer:b_tvels       offset:0 atIndex:1];
    [enc setBuffer:b_CoMs        offset:0 atIndex:2];
    [enc setBuffer:b_velo_prof   offset:0 atIndex:3];
    [enc setBuffer:b_grad_prof   offset:0 atIndex:4];
    [enc setBuffer:b_global_max  offset:0 atIndex:5];
    [enc setBuffer:b_radii       offset:0 atIndex:6];
    NSUInteger gridSz = (NSUInteger)num_particles;
    NSUInteger tgSize = (NSUInteger)ms.TPB;
    [enc dispatchThreads:MTLSizeMake(gridSz,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
    [enc endEncoding];
    commit_wait(cb);
}

// ============================================================
//  High-level simulation functions (Metal-adapted, matching CUDA originals)
// ============================================================

// Read coord nums from buffer and compute average (no GPU copy needed)
float find_avg_coord_num(int num_particles, id<MTLBuffer> b_coord_nums) {
    int *ptr = (int*)b_coord_nums.contents;
    double sum = 0.0;
    for (int i=0; i<num_particles; i++) sum += ptr[i];
    return (float)(sum / num_particles);
}

// Compute avg kinetic energy using GPU kernel
float find_avg_energy(MetalSim &ms, int num_particles,
    id<MTLBuffer> b_masses, id<MTLBuffer> b_tvels,
    id<MTLBuffer> b_moi, id<MTLBuffer> b_avels, id<MTLBuffer> b_KE)
{
    dispatch_find_energy_kernel(ms, num_particles, b_masses, b_tvels, b_moi, b_avels, b_KE);
    float *ptr = (float*)b_KE.contents;
    double sum = 0.0;
    for (int i=0; i<num_particles; i++) sum += ptr[i];
    return (float)(sum / num_particles);
}

void reset_stress_profiles(int num_bins,
    id<MTLBuffer> b_vol_frac, id<MTLBuffer> b_cn_prof,
    id<MTLBuffer> b_ct_prof,  id<MTLBuffer> b_l_prof)
{
    memset(b_vol_frac.contents, 0, num_bins * sizeof(float));
    memset(b_cn_prof.contents,  0, num_bins * sizeof(float));
    memset(b_ct_prof.contents,  0, num_bins * sizeof(float));
    memset(b_l_prof.contents,   0, num_bins * sizeof(float));
}

void reset_time_scales(id<MTLBuffer> b_cont, id<MTLBuffer> b_force, id<MTLBuffer> b_torque) {
    float init_val = 1e6f;
    *(float*)b_cont.contents   = init_val;
    *(float*)b_force.contents  = init_val;
    *(float*)b_torque.contents = init_val;
}

// reset_particles_for_new_time_step  (uses explicit buffers for relaxation compatibility)
void reset_particles_for_new_time_step_ex(
    MetalSim &ms, int num_particles, long long combis, bool gravity, float3 grav_dir,
    bool friction_toggle,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_cn, id<MTLBuffer> b_ct, id<MTLBuffer> b_l,
    id<MTLBuffer> b_masses, id<MTLBuffer> b_coord_nums,
    id<MTLBuffer> b_num_ref, id<MTLBuffer> b_old_int, id<MTLBuffer> b_new_int)
{
    dispatch_clear_kernel(ms, num_particles, gravity, grav_dir,
        b_forces, b_torques, b_cn, b_ct, b_l, b_masses, b_coord_nums);
    if (friction_toggle)
        dispatch_clear_interactions_kernel(ms, num_particles, combis, b_num_ref, b_old_int, b_new_int);
}

// Convenience wrapper using main simulation buffers
void reset_particles_for_new_time_step(MetalSim &ms, bool gravity, float3 grav_dir, bool friction_toggle) {
    reset_particles_for_new_time_step_ex(ms, ms.num_particles, ms.combis, gravity, grav_dir, friction_toggle,
        ms.buf_forces, ms.buf_torques, ms.buf_cn_stresses, ms.buf_ct_stresses, ms.buf_l_stresses,
        ms.buf_masses, ms.buf_coord_nums, ms.buf_num_particles_ref, ms.buf_old_int, ms.buf_new_int);
}

// run_pairwise_interactions with explicit buffers (for relaxation)
void run_pairwise_interactions_ex(
    MetalSim &ms, int num_particles,
    float dt, float viscosity, float min_sep, float max_sep,
    float ee_w, float ss_w, float es_w,
    int contact_toggle, int friction_toggle, int lub_toggle,
    float3 sys_dim, int3 pb, float LEBC_shift, float LEBC_velo, bool gen_phase,
    id<MTLBuffer> b_CoMs, id<MTLBuffer> b_oris, id<MTLBuffer> b_shafts,
    id<MTLBuffer> b_radii, id<MTLBuffer> b_masses, id<MTLBuffer> b_avg_inerts,
    id<MTLBuffer> b_coord_nums, id<MTLBuffer> b_fric,
    id<MTLBuffer> b_tvels, id<MTLBuffer> b_avels,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_cn, id<MTLBuffer> b_ct, id<MTLBuffer> b_l,
    id<MTLBuffer> b_old_int, id<MTLBuffer> b_new_int, id<MTLBuffer> b_moi,
    id<MTLBuffer> b_min_dt_cont, id<MTLBuffer> b_min_dt_force, id<MTLBuffer> b_min_dt_torque,
    id<MTLBuffer> b_pair_consts)
{
    dispatch_pair_kernel(ms, num_particles, dt, viscosity, min_sep, max_sep,
        ee_w, ss_w, es_w, contact_toggle, friction_toggle, lub_toggle,
        sys_dim, pb, LEBC_shift, LEBC_velo, gen_phase,
        b_CoMs, b_oris, b_shafts, b_radii, b_masses, b_avg_inerts, b_coord_nums,
        b_fric, b_tvels, b_avels,
        b_forces, b_torques, b_cn, b_ct, b_l, b_old_int, b_new_int, b_moi,
        b_min_dt_cont, b_min_dt_force, b_min_dt_torque, b_pair_consts);
}

// Convenience wrapper using main simulation buffers
void run_pairwise_interactions(
    MetalSim &ms, float dt, float viscosity, float min_sep, float max_sep,
    float ee_w, float ss_w, float es_w,
    int contact_toggle, int friction_toggle, int lub_toggle,
    float3 sys_dim, int3 pb, float LEBC_shift, float LEBC_velo, bool gen_phase=false)
{
    run_pairwise_interactions_ex(ms, ms.num_particles, dt, viscosity, min_sep, max_sep,
        ee_w, ss_w, es_w, contact_toggle, friction_toggle, lub_toggle,
        sys_dim, pb, LEBC_shift, LEBC_velo, gen_phase,
        ms.buf_CoMs, ms.buf_oris, ms.buf_shafts, ms.buf_radii, ms.buf_masses, ms.buf_avg_inerts,
        ms.buf_coord_nums, ms.buf_fric_coefs,
        ms.buf_tvels, ms.buf_avels, ms.buf_forces, ms.buf_torques,
        ms.buf_cn_stresses, ms.buf_ct_stresses, ms.buf_l_stresses,
        ms.buf_old_int, ms.buf_new_int, ms.buf_moi,
        ms.buf_min_dt_cont, ms.buf_min_dt_force, ms.buf_min_dt_torque,
        ms.buf_pair_consts);
}

// run_single_body_interactions with explicit buffers
void run_single_body_interactions_ex(
    MetalSim &ms, int num_particles, float3 sys_dim, int3 pb,
    int contact_toggle, int drag_toggle, int lift_toggle,
    int num_bins, float bin_size, float viscosity, float fluid_density, float max_height,
    bool gen_phase,
    id<MTLBuffer> b_CoMs, id<MTLBuffer> b_oris,
    id<MTLBuffer> b_ep1, id<MTLBuffer> b_ep2,
    id<MTLBuffer> b_radii, id<MTLBuffer> b_shafts, id<MTLBuffer> b_masses,
    id<MTLBuffer> b_kns, id<MTLBuffer> b_ens,
    id<MTLBuffer> b_tvels, id<MTLBuffer> b_avels,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_velo, id<MTLBuffer> b_grad,
    id<MTLBuffer> b_coord_nums)
{
    dispatch_body_kernel(ms, num_particles, pb, sys_dim,
        contact_toggle, drag_toggle, lift_toggle,
        num_bins, bin_size, viscosity, fluid_density, max_height, gen_phase,
        b_CoMs, b_oris, b_ep1, b_ep2, b_radii, b_shafts, b_masses,
        b_kns, b_ens, b_tvels, b_avels, b_forces, b_torques,
        b_velo, b_grad, b_coord_nums);
}

// Convenience wrapper
void run_single_body_interactions(
    MetalSim &ms, float3 sys_dim, int3 pb,
    int contact_toggle, int drag_toggle, int lift_toggle,
    int num_bins, float bin_size, float viscosity, float fluid_density, float max_height,
    bool gen_phase=false)
{
    run_single_body_interactions_ex(ms, ms.num_particles, sys_dim, pb,
        contact_toggle, drag_toggle, lift_toggle,
        num_bins, bin_size, viscosity, fluid_density, max_height, gen_phase,
        ms.buf_CoMs, ms.buf_oris, ms.buf_endpoints1, ms.buf_endpoints2,
        ms.buf_radii, ms.buf_shafts, ms.buf_masses, ms.buf_kns, ms.buf_ens,
        ms.buf_tvels, ms.buf_avels, ms.buf_forces, ms.buf_torques,
        ms.buf_velo_profile, ms.buf_grad_profile, ms.buf_coord_nums);
}

// integrate_EoM with explicit buffers
void integrate_EoM_ex(
    MetalSim &ms, int num_particles, float3 sys_dim, int3 pb,
    float LEBC_shift, float LEBC_velo, float dt, bool allowRotation,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_CoMs, id<MTLBuffer> b_tvels,
    id<MTLBuffer> b_taccs, id<MTLBuffer> b_masses, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_oris, id<MTLBuffer> b_ep1, id<MTLBuffer> b_ep2,
    id<MTLBuffer> b_shafts, id<MTLBuffer> b_avels, id<MTLBuffer> b_aaccs, id<MTLBuffer> b_moi)
{
    dispatch_integrate_kernel(ms, num_particles, sys_dim, pb, LEBC_shift, LEBC_velo, dt, allowRotation,
        b_forces, b_CoMs, b_tvels, b_taccs, b_masses, b_torques,
        b_oris, b_ep1, b_ep2, b_shafts, b_avels, b_aaccs, b_moi);
}

// Convenience wrapper
void integrate_EoM(MetalSim &ms, float3 sys_dim, int3 pb,
    float LEBC_shift, float LEBC_velo, float dt, bool allowRotation=true)
{
    integrate_EoM_ex(ms, ms.num_particles, sys_dim, pb, LEBC_shift, LEBC_velo, dt, allowRotation,
        ms.buf_forces, ms.buf_CoMs, ms.buf_tvels, ms.buf_taccs, ms.buf_masses, ms.buf_torques,
        ms.buf_oris, ms.buf_endpoints1, ms.buf_endpoints2, ms.buf_shafts,
        ms.buf_avels, ms.buf_aaccs, ms.buf_moi);
}

// find_new_dt — reads from buffer .contents (no GPU dispatch needed here)
void find_new_dt(float &dt, const bool dynamic_dt,
    id<MTLBuffer> b_min_dt_cont, id<MTLBuffer> b_min_dt_force, id<MTLBuffer> b_min_dt_torque,
    const float *fluid_velocity_gradient_profile,
    const float max_dt, const float min_dt, const float dyn_dt_scale, Helper &helper)
{
    if (helper.total_time < 0.1f) return;
    if (dynamic_dt) {
        float min_dt_cont   = *(float*)b_min_dt_cont.contents;
        float min_dt_force  = *(float*)b_min_dt_force.contents;
        float min_dt_torque = *(float*)b_min_dt_torque.contents;
        float velocity_scale = fluid_velocity_gradient_profile[0];
        min_dt_force  = fabsf(min_dt_force  * velocity_scale);
        min_dt_torque = fabsf(min_dt_torque * velocity_scale);
        double min_min_dt = fminf(fminf(min_dt_cont, min_dt_force), min_dt_torque);
        float new_dt = fminf(max_dt, fmaxf(min_dt, (float)min_min_dt / dyn_dt_scale));
        dt = new_dt;
    }
}

// shrink_system
void shrink_system(MetalSim &ms, float3 &system_dimensions, float3 target_dimensions,
    float delta_vf_step_size, float total_rods_vol, id<MTLBuffer> b_CoMs, int num_particles)
{
    float V_old = system_dimensions.x * system_dimensions.y * system_dimensions.z;
    float vf_old = total_rods_vol / V_old;
    float vf_new = vf_old + delta_vf_step_size;
    if (vf_new >= 0.999999f) vf_new = 0.999999f;
    float V_new = total_rods_vol / vf_new;
    float scale = cbrtf(V_new / V_old);
    float3 sys2 = system_dimensions;
    system_dimensions.x = fmaxf(system_dimensions.x * scale, target_dimensions.x);
    system_dimensions.y = fmaxf(system_dimensions.y * scale, target_dimensions.y);
    system_dimensions.z = fmaxf(system_dimensions.z * scale, target_dimensions.z);
    scale = system_dimensions.x / sys2.x;
    dispatch_scale_kernel(ms, num_particles, scale, b_CoMs);
}

// find_new_profiles
void find_new_profiles(MetalSim &ms, bool fixed_stress,
    int num_particles, float3 sys_dim, int3 pb,
    int num_bins, float bin_size,
    float *vol_frac_profile, float *cn_stress_profile, float *ct_stress_profile, float *l_stress_profile,
    float *fluid_velocity_profile, int num_velos_specified, float *fluid_velocity_gradient_profile,
    float aspect, float fluid_viscosity,
    float *fluid_stress_profile, float *shear_stress_profile,
    float *fixed_stress_profile, float controller_gain)
{
    if (fixed_stress) {
        dispatch_get_profiles_kernel(ms, num_particles, num_bins, bin_size, sys_dim, pb,
            ms.buf_CoMs, ms.buf_endpoints1, ms.buf_endpoints2, ms.buf_volumes,
            ms.buf_vol_frac_profile, ms.buf_cn_stresses, ms.buf_ct_stresses, ms.buf_l_stresses,
            ms.buf_cn_stress_profile, ms.buf_ct_stress_profile, ms.buf_l_stress_profile);

        // read back via .contents
        memcpy(vol_frac_profile,  ms.buf_vol_frac_profile.contents,     num_bins*sizeof(float));
        memcpy(cn_stress_profile, ms.buf_cn_stress_profile.contents,    num_bins*sizeof(float));
        memcpy(ct_stress_profile, ms.buf_ct_stress_profile.contents,    num_bins*sizeof(float));
        memcpy(l_stress_profile,  ms.buf_l_stress_profile.contents,     num_bins*sizeof(float));
        memcpy(fluid_velocity_profile,          ms.buf_velo_profile.contents, num_velos_specified*sizeof(float));
        memcpy(fluid_velocity_gradient_profile, ms.buf_grad_profile.contents, num_bins*sizeof(float));

        fluid_stress(vol_frac_profile, aspect, fluid_viscosity, fluid_velocity_gradient_profile, (float)num_bins, fluid_stress_profile);
        shear_stress((float)num_bins, cn_stress_profile, ct_stress_profile, l_stress_profile, fluid_stress_profile, shear_stress_profile);
        stress_controller(fixed_stress_profile, shear_stress_profile, fluid_velocity_gradient_profile, num_bins, controller_gain);
        velocity_profile_eval(fluid_velocity_profile, num_bins, bin_size, fluid_velocity_gradient_profile);

        memcpy(ms.buf_velo_profile.contents, fluid_velocity_profile,          num_velos_specified*sizeof(float));
        memcpy(ms.buf_grad_profile.contents, fluid_velocity_gradient_profile, num_bins*sizeof(float));
    }
}


// save_sequency — Metal version (reads from .contents instead of cudaMemcpy)
void save_sequency(
    bool doSave, bool doPrint, bool fixed_stress,
    MetalSim &ms, int num_particles, long long combis,
    float3 sys_dim, int3 pb,
    int num_bins, float bin_size,
    float *vol_frac_profile, float *cn_stress_profile, float *ct_stress_profile,
    float *l_stress_profile, float *fluid_velocity_profile,
    int num_velos_specified, float *fluid_velocity_gradient_profile,
    float aspect, float fluid_viscosity, int time_step,
    float *radii, float *shaft_lengths, std::string folder_path, int &frame,
    float tot_strain, float dt, float fluid_density, float max_height,
    std::string fluid_info_folder_path,
    float *fluid_stress_profile, float *shear_stress_profile,
    float3 n_ref, float tot_time, float dt_avg,
    OutputFlags output_flags)
{
    if (!doSave) return;

    if (!fixed_stress) {
        dispatch_get_profiles_kernel(ms, num_particles, num_bins, bin_size, sys_dim, pb,
            ms.buf_CoMs, ms.buf_endpoints1, ms.buf_endpoints2, ms.buf_volumes,
            ms.buf_vol_frac_profile, ms.buf_cn_stresses, ms.buf_ct_stresses, ms.buf_l_stresses,
            ms.buf_cn_stress_profile, ms.buf_ct_stress_profile, ms.buf_l_stress_profile);

        memcpy(vol_frac_profile,  ms.buf_vol_frac_profile.contents,     num_bins*sizeof(float));
        memcpy(cn_stress_profile, ms.buf_cn_stress_profile.contents,    num_bins*sizeof(float));
        memcpy(ct_stress_profile, ms.buf_ct_stress_profile.contents,    num_bins*sizeof(float));
        memcpy(l_stress_profile,  ms.buf_l_stress_profile.contents,     num_bins*sizeof(float));
        memcpy(fluid_velocity_profile, ms.buf_velo_profile.contents, num_velos_specified*sizeof(float));
        memcpy(fluid_velocity_gradient_profile, ms.buf_grad_profile.contents, num_bins*sizeof(float));
    }

    // CPU pointers are just .contents for unified memory
    float3  *CoMs       = (float3*)ms.buf_CoMs.contents;
    float3  *oris       = (float3*)ms.buf_oris.contents;
    float3  *tvelos     = (float3*)ms.buf_tvels.contents;
    float3  *avelos     = (float3*)ms.buf_avels.contents;
    float3  *forces     = (float3*)ms.buf_forces.contents;
    float3  *torques    = (float3*)ms.buf_torques.contents;
    mat33   *cn_stress  = (mat33*)ms.buf_cn_stresses.contents;
    mat33   *ct_stress  = (mat33*)ms.buf_ct_stresses.contents;
    mat33   *lub_stress = (mat33*)ms.buf_l_stresses.contents;
    int     *coord_nums = (int*)ms.buf_coord_nums.contents;

    fluid_stress(vol_frac_profile, aspect, fluid_viscosity, fluid_velocity_gradient_profile, (float)num_bins, fluid_stress_profile);
    shear_stress((float)num_bins, cn_stress_profile, ct_stress_profile, l_stress_profile, fluid_stress_profile, shear_stress_profile);

    float cn_visc{}, ct_visc{}, lub_visc{}, fluid_visc{}, total_visc{};
    convert_to_avg_viscosity(cn_stress_profile,    fluid_viscosity, fluid_velocity_gradient_profile, num_bins, cn_visc);
    convert_to_avg_viscosity(ct_stress_profile,    fluid_viscosity, fluid_velocity_gradient_profile, num_bins, ct_visc);
    convert_to_avg_viscosity(l_stress_profile,     fluid_viscosity, fluid_velocity_gradient_profile, num_bins, lub_visc);
    convert_to_avg_viscosity(fluid_stress_profile, fluid_viscosity, fluid_velocity_gradient_profile, num_bins, fluid_visc);
    convert_to_avg_viscosity(shear_stress_profile, fluid_viscosity, fluid_velocity_gradient_profile, num_bins, total_visc);

    float avg_stress{}, avg_shearrate{};
    avg_array(shear_stress_profile, avg_stress, num_bins);
    avg_array(fluid_velocity_gradient_profile, avg_shearrate, num_bins);

    float S{}, S_ref{};
    float3 director{};
    find_order_parameters(num_particles, oris, n_ref, S, director, S_ref);

    float avg_coord_num = find_avg_coord_num(num_particles, ms.buf_coord_nums);

    float raw_dt_cont  = *(float*)ms.buf_min_dt_cont.contents;
    float raw_dt_force = *(float*)ms.buf_min_dt_force.contents;
    float raw_dt_torq  = *(float*)ms.buf_min_dt_torque.contents;

    std::vector<double> order_output = {
        tot_time, tot_strain, total_visc, cn_visc, ct_visc, lub_visc, fluid_visc,
        S_ref, S, director.x, director.y, director.z, avg_stress, avg_shearrate, avg_coord_num,
        dt, raw_dt_cont, raw_dt_force, raw_dt_torq
    };

    if (output_flags.doSimpleOut) {
        writeCSV(folder_path, "simple_output.csv", order_output);
        if (doPrint) {
            std::cout << std::setprecision(6);
            for (size_t i = 0; i < order_output.size(); ++i) {
                std::cout << order_output[i];
                if (i < order_output.size()-1) std::cout << ",";
            }
            std::cout << "\n\n" << std::flush;
        }
    }

    if (output_flags.doVisOut || frame==0)
        to_lammps_dump(num_particles, time_step, sys_dim, CoMs, oris, radii, shaft_lengths,
            tvelos, avelos, forces, torques, cn_stress, ct_stress, lub_stress, coord_nums, folder_path);

    if (output_flags.doFluidOut)
        fluid_info_output(frame, tot_strain, tot_time, dt, dt_avg, fluid_viscosity, fluid_density,
            max_height, num_bins, fluid_velocity_profile, fluid_velocity_gradient_profile,
            shear_stress_profile, cn_stress_profile, ct_stress_profile, l_stress_profile,
            fluid_stress_profile, vol_frac_profile, true, S, director, S_ref, n_ref,
            fluid_info_folder_path);

    frame += 1;
}

// checkpoint_sequence — Metal version (reads from .contents)
void checkpoint_sequence(
    bool doFull_info, MetalSim &ms, int num_particles, long long combis,
    std::string rod_info_folder_path, int time_step, float LEBC_shift,
    int *ids, float *radii, float *aspects, float *shaft_lengths,
    float *densities, float *volumes, float *masses,
    float *avg_inertias, float *kns, float *ens, float *ets, float *fric_coefs,
    int frame, OutputFlags output_flags)
{
    if (!doFull_info) return;
    // All data lives in shared buffers — read directly
    float3 *CoMs          = (float3*)ms.buf_CoMs.contents;
    float3 *oris          = (float3*)ms.buf_oris.contents;
    float3 *endpoints1    = (float3*)ms.buf_endpoints1.contents;
    float3 *endpoints2    = (float3*)ms.buf_endpoints2.contents;
    float3 *tvelos        = (float3*)ms.buf_tvels.contents;
    float3 *avelos        = (float3*)ms.buf_avels.contents;
    float3 *forces        = (float3*)ms.buf_forces.contents;
    float3 *torques       = (float3*)ms.buf_torques.contents;
    float3 *accelerations = (float3*)ms.buf_taccs.contents;
    float3 *ang_accs      = (float3*)ms.buf_aaccs.contents;
    float3 *moi           = (float3*)ms.buf_moi.contents;
    float3 *old_int       = (float3*)ms.buf_old_int.contents;
    float3 *new_int       = (float3*)ms.buf_old_int.contents;  // old_int holds current values in-place

    if (output_flags.doCheckpointOut) {
        std::string fname = rod_info_folder_path + std::to_string(frame) + "_full_info.bin";
        write_particles_binary(fname, num_particles, LEBC_shift, ids, radii, aspects, shaft_lengths,
            densities, volumes, masses, moi, avg_inertias, kns, ens, ets, fric_coefs,
            CoMs, oris, endpoints1, endpoints2, tvelos, avelos, accelerations, ang_accs,
            forces, torques, old_int, new_int, combis);
    }
}

// ============================================================
// PART 4a: CPU particle generation functions
// ============================================================

struct ParticleRow { int id; float r; float len; float density; float kn; };

void load_particle_list_simple(const std::string &filename, std::vector<ParticleRow> &out_rows, float &out_vol_frac)
{
    out_rows.clear();
    std::ifstream ifs(filename);
    std::string line;
    while (std::getline(ifs, line)) {
        auto p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos) continue;
        line = line.substr(p);
        if (line.rfind("vol_frac", 0) == 0) {
            auto eq = line.find('=');
            if (eq != std::string::npos) out_vol_frac = std::stof(line.substr(eq+1));
            continue;
        }
        bool has_alpha = false;
        for (char c : line) if (std::isalpha((unsigned char)c)) { has_alpha = true; break; }
        if (has_alpha) continue;
        std::replace(line.begin(), line.end(), '\t', ',');
        std::replace(line.begin(), line.end(), ' ', ',');
        std::stringstream ss(line);
        std::string a,b,c,d,e;
        if (!std::getline(ss,a,',')) continue;
        if (!std::getline(ss,b,',')) continue;
        if (!std::getline(ss,c,',')) continue;
        if (!std::getline(ss,d,',')) continue;
        if (!std::getline(ss,e,',')) continue;
        ParticleRow r; r.id = std::stoi(a); r.r = std::stof(b); r.len = std::stof(c);
        r.density = std::stof(d); r.kn = std::stof(e);
        out_rows.push_back(r);
    }
}

void generate_particles_from_polyfile(
    int &num_particles,
    float density, float kn, float en, float et, float friction_coef,
    float max_tvelo, float max_avelo,
    float3 &system_dimensions, float3 &target_dimensions,
    int *ids, float *radii, float *aspects, float *shaft_lengths, float *densities,
    float *volumes, float *masses, float3 *moments_of_inertia, float *avg_inertias,
    int *coord_nums, float *kns, float *ens, float *ets, float *fric_coefs,
    float h_max,
    float3 *CoMs, float3 *oris, float3 *endpoints1, float3 *endpoints2,
    float3 *tvelos, float3 *avelos,
    float3 *forces, float3 *torques, float3 *accelerations, float3 *angular_accelerations,
    float3 *old_interactions, float3 *new_interactions, long long &combis, float &vol_frac,
    float oop, float3 ref_direction,
    float intial_vf, float &total_rods_vol,
    const std::string &particle_list_file)
{
    std::vector<ParticleRow> rows;
    float file_vf = vol_frac;
    load_particle_list_simple(particle_list_file, rows, file_vf);
    vol_frac = file_vf;
    num_particles = (int)rows.size();
    combis = upper_tri_buf_size(num_particles);
    total_rods_vol = 0.0f;
    float original_vol = system_dimensions.x * system_dimensions.y * system_dimensions.z;

    for (int i = 0; i < num_particles; ++i) {
        const ParticleRow &rw = rows[i];
        ids[i] = rw.id;
        radii[i] = rw.r;
        float asp = rw.len / (2 * rw.r);
        if (asp <= 3){ std::cout << "!WARNING! aspect < 3 !WARNING!\n"; }
        aspects[i] = asp;
        float shaft_length = rw.len - (2 * rw.r);
        shaft_lengths[i] = shaft_length;
        float vol = (4.0f/3.0f * h_PI * powf(rw.r, 3.0f)) + h_PI * shaft_length * powf(rw.r, 2.0f);
        volumes[i] = vol; densities[i] = rw.density; masses[i] = vol * rw.density;
        float inertia_xx = h_PI * rw.density * (1.0f/12.0f * powf(rw.r, 2.0f) * powf(shaft_length, 3.0f)
                            + 8.0f/15.0f * powf(rw.r, 5.0f)
                            + 1.0f/3.0f * powf(rw.r, 3.0f) * powf(shaft_length, 2.0f)
                            + 3.0f/4.0f * powf(rw.r, 4.0f) * shaft_length);
        float inertia_zz = h_PI * rw.density * (0.5f * powf(rw.r, 4.0f) * shaft_length + 8.0f/15.0f * powf(rw.r, 5.0f));
        moments_of_inertia[i] = make_float3(inertia_xx, inertia_xx, inertia_zz);
        avg_inertias[i] = (inertia_xx + inertia_xx + inertia_zz) / 3.0f;
        kns[i] = rw.kn; ens[i] = en; ets[i] = et; fric_coefs[i] = friction_coef;
        total_rods_vol += vol;
    }

    float target_scale_factor = powf((total_rods_vol / vol_frac) / original_vol, 1.0f/3.0f);
    float intial_scale_factor = powf((total_rods_vol / intial_vf) / original_vol, 1.0f/3.0f);
    target_dimensions.x = system_dimensions.x * target_scale_factor;
    target_dimensions.y = system_dimensions.y * target_scale_factor;
    target_dimensions.z = system_dimensions.z * target_scale_factor;
    system_dimensions.x *= intial_scale_factor;
    system_dimensions.y *= intial_scale_factor;
    system_dimensions.z *= intial_scale_factor;

    for (int i = 0; i < num_particles; ++i) {
        float3 cm = make_float3(rand01() * system_dimensions.x,
                                rand01() * system_dimensions.y,
                                rand01() * system_dimensions.z);
        CoMs[i] = cm;
        float3 orientation = normalize(make_float3(rand_m1_to_1(), rand_m1_to_1(), rand_m1_to_1()));
        double w = find_w_for_S(oop);
        float3 refd = normalize(ref_direction);
        float3 v = refd * (float)w + orientation * (1.0f - (float)w);
        oris[i] = normalize(v);
        float shaft_length = shaft_lengths[i];
        endpoints1[i] = CoMs[i] + oris[i] * (shaft_length / 2.0f);
        endpoints2[i] = CoMs[i] - oris[i] * (shaft_length / 2.0f);
        tvelos[i] = make_float3(rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo);
        avelos[i] = make_float3(rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo);
        forces[i] = make_float3(0,0,0); torques[i] = make_float3(0,0,0);
        accelerations[i] = make_float3(0,0,0); angular_accelerations[i] = make_float3(0,0,0);
        coord_nums[i] = 0;
    }
    for (long long i = 0; i < combis; ++i) {
        old_interactions[i] = make_float3(0,0,0);
        new_interactions[i] = make_float3(0,0,0);
    }
    std::cout << "Loaded " << num_particles << " particles from " << particle_list_file << "\n";
    std::cout << "Total rods volume = " << total_rods_vol << ", target vol_frac = " << vol_frac << "\n";
}

void generate_particles(int &num_particles, float radius_1, float radius_2, float prop_1, float prop_2,
    float aspect, float density, float kn, float en, float et, float friction_coef,
    float max_tvelo, float max_avelo, float3 &system_dimensions,
    int *ids, float *radii, float *aspects, float *shaft_lengths, float *densities,
    float *volumes, float *masses, float3 *moments_of_inertia, float *avg_inertias, int *coord_nums,
    float *kns, float *ens, float *ets, float *fric_coefs,
    float h_max,
    float3 *CoMs, float3 *oris, float3 *endpoints1, float3 *endpoints2,
    float3 *tvelos, float3 *avelos,
    float3 *forces, float3 *torques, float3 *accelerations, float3 *angular_accelerations,
    float3 *old_interactions, float3 *new_interactions, long long &combis,
    bool safe_generation, bool set_vol_frac, bool fixed_system_size, float vol_frac,
    float oop, float3 ref_direction,
    bool load_system, std::string load_data_location, int starting_time_step,
    bool positions_only, bool continue_sim)
{
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Rods Generating... \n";
    std::cout << "--------------------------------------------------------\n";

    int particle_types = 2;

    if (!load_system) {
        prop_1 = prop_1 / (prop_1 + prop_2);
        prop_2 = 1.0f - prop_1;

        int num_particle_1{}, num_particle_2{};

        if (set_vol_frac) {
            float min_dimensions = 4 * fmaxf(radius_1, radius_2) * (aspect + h_max);
            float rod_vol_1 = (4.0f/3.0f * h_PI * powf(radius_1, 3.0f) + h_PI * 2.0f * radius_1 * (aspect - 1.0f) * radius_1 * radius_1);
            float rod_vol_2 = (4.0f/3.0f * h_PI * powf(radius_2, 3.0f) + h_PI * 2.0f * radius_2 * (aspect - 1.0f) * radius_2 * radius_2);
            float rel_vol_1 = rod_vol_1 * prop_1;
            float rel_vol_2 = rod_vol_2 * prop_2;

            if (fixed_system_size) {
                float sys_volume = system_dimensions.x * system_dimensions.y * system_dimensions.z;
                float aim_tot_rod_vol = vol_frac * sys_volume;
                num_particles = int(roundf(aim_tot_rod_vol/(rel_vol_1 + rel_vol_2)));
                num_particle_1 = num_particles * prop_1;
                num_particle_2 = num_particles * prop_2;
                num_particles = num_particle_1 + num_particle_2;
            } else {
                num_particle_1 = num_particles * prop_1;
                num_particle_2 = num_particles * prop_2;
                num_particles = num_particle_1 + num_particle_2;
                float aim_tot_rod_vol = (num_particle_1 * rod_vol_1) + (num_particle_2 * rod_vol_2);
                float sys_vol = aim_tot_rod_vol / vol_frac;
                float original_vol = system_dimensions.x * system_dimensions.y * system_dimensions.z;
                float scale_factor = powf(sys_vol / original_vol, 1.0f/3.0f);
                if (safe_generation) {
                    system_dimensions.x = fmaxf(system_dimensions.x * scale_factor, min_dimensions);
                    system_dimensions.y = fmaxf(system_dimensions.y * scale_factor, min_dimensions);
                    system_dimensions.z = fmaxf(system_dimensions.z * scale_factor, min_dimensions);
                    float sys_volume = system_dimensions.x * system_dimensions.y * system_dimensions.z;
                    aim_tot_rod_vol = vol_frac * sys_volume;
                    num_particles = int(roundf(aim_tot_rod_vol/(rel_vol_1 + rel_vol_2)));
                    num_particle_1 = num_particles * prop_1;
                    num_particle_2 = num_particles * prop_2;
                    num_particles = num_particle_1 + num_particle_2;
                } else {
                    system_dimensions.x = system_dimensions.x * scale_factor;
                    system_dimensions.y = system_dimensions.y * scale_factor;
                    system_dimensions.z = system_dimensions.z * scale_factor;
                }
            }
        } else {
            num_particle_1 = num_particles * prop_1;
            num_particle_2 = num_particles * prop_2;
            num_particles = num_particle_1 + num_particle_2;
        }

        float radiuses[2] = { radius_1, radius_2 };
        int part_nums[2] = { num_particle_1, num_particle_2 };
        int idx = 0;
        for (int j = 0; j < particle_types; ++j) {
            for (int k = 0; k < part_nums[j]; ++k) {
                int i = idx++;
                ids[i] = i;
                radii[i] = radiuses[j];
                aspects[i] = aspect;
                float shaft_length = 2.0f * radiuses[j] * (aspect - 1.0f);
                shaft_lengths[i] = shaft_length;
                float volume = (4.0f/3.0f * h_PI * powf(radiuses[j], 3.0f)) + h_PI * shaft_length * powf(radiuses[j], 2.0f);
                volumes[i] = volume; densities[i] = density; masses[i] = volume * density;
                float inertia_xx = h_PI * density * (1.0f/12.0f * powf(radiuses[j], 2.0f) * powf(shaft_length, 3.0f)
                                    + 8.0f/15.0f * powf(radiuses[j], 5.0f)
                                    + 1.0f/3.0f * powf(radiuses[j], 3.0f) * powf(shaft_length, 2.0f)
                                    + 3.0f/4.0f * powf(radiuses[j], 4.0f) * shaft_length);
                float inertia_zz = h_PI * density * (0.5f * powf(radiuses[j], 4.0f) * shaft_length + 8.0f/15.0f * powf(radiuses[j], 5.0f));
                moments_of_inertia[i] = make_float3(inertia_xx, inertia_xx, inertia_zz);
                avg_inertias[i] = (inertia_xx + inertia_xx + inertia_zz) / 3.0f;
                kns[i] = kn; ens[i] = en; ets[i] = et; fric_coefs[i] = friction_coef;

                float3 cm = make_float3(rand01()*system_dimensions.x, rand01()*system_dimensions.y, rand01()*system_dimensions.z);
                CoMs[i] = cm;
                float3 orientation = normalize(make_float3(rand_m1_to_1(), rand_m1_to_1(), rand_m1_to_1()));
                double w = find_w_for_S(oop);
                ref_direction = normalize(ref_direction);
                float3 v = ref_direction * (float)w + orientation * (1.0f - (float)w);
                orientation = normalize(v);
                oris[i] = orientation;
                endpoints1[i] = cm + orientation * (shaft_length / 2.0f);
                endpoints2[i] = cm - orientation * (shaft_length / 2.0f);
                tvelos[i] = make_float3(rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo);
                avelos[i] = make_float3(rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo);
                forces[i] = make_float3(0,0,0); torques[i] = make_float3(0,0,0);
                accelerations[i] = make_float3(0,0,0); angular_accelerations[i] = make_float3(0,0,0);
                coord_nums[i] = 0;
            }
        }

        for (long long i = 0; i < combis; i++) {
            old_interactions[i] = make_float3(0,0,0);
            new_interactions[i] = make_float3(0,0,0);
        }

        float total_particle_vol = 0;
        for (int i = 0; i < num_particles; i++) total_particle_vol += volumes[i];
        float system_vol = system_dimensions.x * system_dimensions.y * system_dimensions.z;
        float actual_vol_Frac = total_particle_vol / system_vol;

        std::cout << num_particles << " particles generated\n";
        std::cout << "Volume fraction: " << std::fixed << std::setprecision(3) << actual_vol_Frac << '\n';
        std::cout << "--------------------------------------------------------\n";
        if (set_vol_frac && !fixed_system_size) {
            std::cout << "System Dimensions Set to: " << std::fixed << std::setprecision(2)
                      << system_dimensions.x << ", " << system_dimensions.y << ", " << system_dimensions.z << '\n';
            std::cout << "--------------------------------------------------------\n";
        }
    } else {
        std::string checkpoint_file = load_data_location + "rod_info/" + std::to_string(starting_time_step) + "_full_info.bin";
        read_particles_binary_into_arrays(checkpoint_file, num_particles, ids,
            radii, aspects, shaft_lengths, densities, volumes, masses, moments_of_inertia, avg_inertias,
            kns, ens, ets, fric_coefs,
            CoMs, oris, endpoints1, endpoints2, tvelos, avelos, forces, torques, accelerations, angular_accelerations,
            old_interactions, new_interactions, combis);

        std::string dump_local = load_data_location + "dump.rod";
        std::array<double,3> L{};
        get_box_size_simple(dump_local, L);
        system_dimensions.x = L[0]; system_dimensions.y = L[1]; system_dimensions.z = L[2];

        if (positions_only) {
            for (int i = 0; i < num_particles; ++i) {
                float mass = volumes[i] * density;
                masses[i] = mass;
                float inertia_xx = h_PI * density * (1.0f/22.0f * powf(radii[i], 2.0f) * powf(shaft_lengths[i], 3.0f)
                                    + 83.0f/240.0f * powf(radii[i], 5.0f)
                                    + 4.0f/3.0f * powf(radii[i], 3.0f) + powf(shaft_lengths[i], 2.0f)
                                    + 3.0f/4.0f * powf(radii[i], 5.0f) + 2.0f * powf(radii[i], 4.0f) * shaft_lengths[i]);
                float inertia_zz = h_PI * density * (0.5f * powf(radii[i], 4.0f) * shaft_lengths[i] + 8.0f/15.0f * powf(radii[i], 5.0f));
                moments_of_inertia[i] = make_float3(inertia_xx, inertia_xx, inertia_zz);
                avg_inertias[i] = (inertia_xx + inertia_xx + inertia_zz) / 3.0f;
                kns[i] = kn; ens[i] = en; ets[i] = et; fric_coefs[i] = friction_coef;
                tvelos[i] = make_float3(rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo);
                avelos[i] = make_float3(rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo);
                forces[i] = make_float3(0,0,0); torques[i] = make_float3(0,0,0);
                accelerations[i] = make_float3(0,0,0); angular_accelerations[i] = make_float3(0,0,0);
                coord_nums[i] = 0;
            }
        }

        float total_particle_vol = 0;
        for (int i = 0; i < num_particles; i++) total_particle_vol += volumes[i];
        float system_vol = system_dimensions.x * system_dimensions.y * system_dimensions.z;
        float actual_vol_Frac = total_particle_vol / system_vol;

        std::cout << "--------------------------------------------------------\n";
        std::cout << num_particles << " particles loaded in\n";
        std::cout << "Volume fraction: " << std::fixed << std::setprecision(3) << actual_vol_Frac << '\n';
        std::cout << "--------------------------------------------------------\n";
    }
}

void generate_particles_with_initial_states(
    int &num_particles,
    float radius, float aspect, float density,
    float kn, float en, float et, float friction_coef,
    float max_tvelo, float max_avelo,
    float3 &system_dimensions,
    int *ids, float *radii, float *aspects, float *shaft_lengths,
    float *densities, float *volumes, float *masses,
    float3 *moments_of_inertia, float *avg_inertias,
    float *kns, float *ens, float *ets, float *fric_coefs,
    float3 *CoMs, float3 *oris, float3 *endpoints1, float3 *endpoints2,
    float3 *tvelos, float3 *avelos,
    float3 *forces, float3 *torques, float3 *accelerations, float3 *angular_accelerations,
    float3 *old_interactions, float3 *new_interactions, long long &combis,
    const float3 *init_CoMs, const float3 *init_oris,
    const float3 *init_tvelos, const float3 *init_avelos)
{
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Rods Generating with supplied initial states... \n";
    std::cout << "--------------------------------------------------------\n";

    float shaft_length = 2.0f * radius * (aspect - 1.0f);

    for (int i = 0; i < num_particles; ++i) {
        ids[i] = i;
        radii[i] = radius;
        aspects[i] = aspect;
        shaft_lengths[i] = shaft_length;

        float volume = (4.0f/3.0f * h_PI * powf(radius, 3.0f)) + h_PI * shaft_length * powf(radius, 2.0f);
        volumes[i] = volume; densities[i] = density; masses[i] = volume * density;

        float inertia_xx = h_PI * density * (1.0f/22.0f * powf(radius, 2.0f) * powf(shaft_length, 3.0f)
                            + 83.0f/240.0f * powf(radius, 5.0f) + 4.0f/3.0f * powf(radius, 3.0f)
                            + powf(shaft_length, 2.0f) + 3.0f/4.0f * powf(radius, 5.0f)
                            + 2.0f * powf(radius, 4.0f) * shaft_length);
        float inertia_zz = h_PI * density * (0.5f * powf(radius, 4.0f) * shaft_length + 8.0f/15.0f * powf(radius, 5.0f));
        moments_of_inertia[i] = make_float3(inertia_xx, inertia_xx, inertia_zz);
        avg_inertias[i] = (inertia_xx + inertia_xx + inertia_zz) / 3.0f;
        kns[i] = kn; ens[i] = en; ets[i] = et; fric_coefs[i] = friction_coef;

        if (init_CoMs != nullptr) { CoMs[i] = init_CoMs[i]; }
        else { CoMs[i] = make_float3(rand01()*system_dimensions.x, rand01()*system_dimensions.y, rand01()*system_dimensions.z); }

        if (init_oris != nullptr) {
            float3 rot = init_oris[i];
            float len = sqrtf(rot.x*rot.x + rot.y*rot.y + rot.z*rot.z);
            oris[i] = (len > 1e-12f) ? make_float3(rot.x/len, rot.y/len, rot.z/len)
                                      : normalize(make_float3(rand_m1_to_1(), rand_m1_to_1(), rand_m1_to_1()));
        } else {
            oris[i] = normalize(make_float3(rand_m1_to_1(), rand_m1_to_1(), rand_m1_to_1()));
        }

        endpoints1[i] = CoMs[i] + oris[i] * (shaft_length / 2.0f);
        endpoints2[i] = CoMs[i] - oris[i] * (shaft_length / 2.0f);

        if (init_tvelos != nullptr) { tvelos[i] = init_tvelos[i]; }
        else { tvelos[i] = make_float3(rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo); }

        if (init_avelos != nullptr) { avelos[i] = init_avelos[i]; }
        else { avelos[i] = make_float3(rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo); }

        forces[i] = make_float3(0,0,0); torques[i] = make_float3(0,0,0);
        accelerations[i] = make_float3(0,0,0); angular_accelerations[i] = make_float3(0,0,0);
    }

    for (long long i = 0; i < combis; ++i) {
        old_interactions[i] = make_float3(0,0,0);
        new_interactions[i] = make_float3(0,0,0);
    }

    float total_particle_vol = 0.0f;
    for (int i = 0; i < num_particles; ++i) total_particle_vol += volumes[i];
    float system_vol = system_dimensions.x * system_dimensions.y * system_dimensions.z;
    std::cout << num_particles << " particles initialised (with user-supplied states where provided)\n";
    std::cout << "Volume fraction: " << std::fixed << std::setprecision(3) << (total_particle_vol/system_vol) << '\n';
    std::cout << "--------------------------------------------------------\n";
}

void generate_particles3(int &num_particles, float radius_1, float radius_2, float prop_1, float prop_2,
    float aspect, float density, float kn, float en, float et, float friction_coef,
    float max_tvelo, float max_avelo, float3 &system_dimensions, float3 &target_dimensions,
    int *ids, float *radii, float *aspects, float *shaft_lengths, float *densities,
    float *volumes, float *masses, float3 *moments_of_inertia, float *avg_inertias, int *coord_nums,
    float *kns, float *ens, float *ets, float *fric_coefs,
    float h_max,
    float3 *CoMs, float3 *oris, float3 *endpoints1, float3 *endpoints2,
    float3 *tvelos, float3 *avelos,
    float3 *forces, float3 *torques, float3 *accelerations, float3 *angular_accelerations,
    float3 *old_interactions, float3 *new_interactions, long long &combis,
    bool safe_generation, bool set_vol_frac, bool fixed_system_size, float vol_frac,
    float oop, float3 ref_direction,
    bool load_system, std::string load_data_location, int starting_time_step,
    bool positions_only, bool continue_sim,
    float intial_vf, float &total_rods_vol)
{
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Rods Generating... \n";
    std::cout << "--------------------------------------------------------\n";

    int particle_types = 2;

    if (!load_system) {
        intial_vf = fminf(intial_vf, vol_frac);
        prop_1 = prop_1 / (prop_1 + prop_2);
        prop_2 = 1.0f - prop_1;

        int num_particle_1{}, num_particle_2{};
        float min_dimensions = 4 * fmaxf(radius_1, radius_2) * (aspect + h_max);
        float rod_vol_1 = (4.0f/3.0f * h_PI * powf(radius_1, 3.0f) + h_PI * 2.0f * radius_1 * (aspect - 1.0f) * radius_1 * radius_1);
        float rod_vol_2 = (4.0f/3.0f * h_PI * powf(radius_2, 3.0f) + h_PI * 2.0f * radius_2 * (aspect - 1.0f) * radius_2 * radius_2);
        float rel_vol_1 = rod_vol_1 * prop_1;
        float rel_vol_2 = rod_vol_2 * prop_2;

        num_particle_1 = num_particles * prop_1;
        num_particle_2 = num_particles * prop_2;
        num_particles = num_particle_1 + num_particle_2;

        float aim_tot_rod_vol = (num_particle_1 * rod_vol_1) + (num_particle_2 * rod_vol_2);
        float original_vol = system_dimensions.x * system_dimensions.y * system_dimensions.z;
        float target_scale_factor = powf((aim_tot_rod_vol / vol_frac) / original_vol, 1.0f/3.0f);
        float intial_scale_factor = powf((aim_tot_rod_vol / intial_vf) / original_vol, 1.0f/3.0f);

        if (safe_generation) {
            target_dimensions.x = fmaxf(system_dimensions.x * target_scale_factor, min_dimensions);
            target_dimensions.y = fmaxf(system_dimensions.y * target_scale_factor, min_dimensions);
            target_dimensions.z = fmaxf(system_dimensions.z * target_scale_factor, min_dimensions);
            float sys_volume = target_dimensions.x * target_dimensions.y * target_dimensions.z;
            float aim_tot2 = vol_frac * sys_volume;
            num_particles = int(roundf(aim_tot2/(rel_vol_1 + rel_vol_2)));
            num_particle_1 = num_particles * prop_1;
            num_particle_2 = num_particles * prop_2;
            num_particles = num_particle_1 + num_particle_2;
        } else {
            target_dimensions.x = system_dimensions.x * target_scale_factor;
            target_dimensions.y = system_dimensions.y * target_scale_factor;
            target_dimensions.z = system_dimensions.z * target_scale_factor;
        }

        system_dimensions.x = system_dimensions.x * intial_scale_factor;
        system_dimensions.y = system_dimensions.y * intial_scale_factor;
        system_dimensions.z = system_dimensions.z * intial_scale_factor;

        float radiuses[2] = { radius_1, radius_2 };
        int part_nums[2] = { num_particle_1, num_particle_2 };
        int idx = 0;
        for (int j = 0; j < particle_types; ++j) {
            for (int k = 0; k < part_nums[j]; ++k) {
                int i = idx++;
                ids[i] = i;
                radii[i] = radiuses[j];
                aspects[i] = aspect;
                float shaft_length = 2.0f * radiuses[j] * (aspect - 1.0f);
                shaft_lengths[i] = shaft_length;
                float volume = (4.0f/3.0f * h_PI * powf(radiuses[j], 3.0f)) + h_PI * shaft_length * powf(radiuses[j], 2.0f);
                volumes[i] = volume; densities[i] = density; masses[i] = volume * density;
                float inertia_xx = h_PI * density * (1.0f/12.0f * powf(radiuses[j], 2.0f) * powf(shaft_length, 3.0f)
                                    + 8.0f/15.0f * powf(radiuses[j], 5.0f)
                                    + 1.0f/3.0f * powf(radiuses[j], 3.0f) * powf(shaft_length, 2.0f)
                                    + 3.0f/4.0f * powf(radiuses[j], 4.0f) * shaft_length);
                float inertia_zz = h_PI * density * (0.5f * powf(radiuses[j], 4.0f) * shaft_length + 8.0f/15.0f * powf(radiuses[j], 5.0f));
                moments_of_inertia[i] = make_float3(inertia_xx, inertia_xx, inertia_zz);
                avg_inertias[i] = (inertia_xx + inertia_xx + inertia_zz) / 3.0f;
                kns[i] = kn; ens[i] = en; ets[i] = et; fric_coefs[i] = friction_coef;

                float3 cm = make_float3(rand01()*system_dimensions.x, rand01()*system_dimensions.y, rand01()*system_dimensions.z);
                CoMs[i] = cm;
                float3 orientation = normalize(make_float3(rand_m1_to_1(), rand_m1_to_1(), rand_m1_to_1()));
                double w = find_w_for_S(oop);
                ref_direction = normalize(ref_direction);
                float3 v = ref_direction * (float)w + orientation * (1.0f - (float)w);
                orientation = normalize(v);
                oris[i] = orientation;
                endpoints1[i] = cm + orientation * (shaft_length / 2.0f);
                endpoints2[i] = cm - orientation * (shaft_length / 2.0f);
                tvelos[i] = make_float3(rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo);
                avelos[i] = make_float3(rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo);
                forces[i] = make_float3(0,0,0); torques[i] = make_float3(0,0,0);
                accelerations[i] = make_float3(0,0,0); angular_accelerations[i] = make_float3(0,0,0);
                coord_nums[i] = 0;
            }
        }

        for (long long i = 0; i < combis; i++) {
            old_interactions[i] = make_float3(0,0,0);
            new_interactions[i] = make_float3(0,0,0);
        }
        for (int i = 0; i < num_particles; i++) total_rods_vol += volumes[i];

    } else {
        std::string checkpoint_file = load_data_location + "rod_info/" + std::to_string(starting_time_step) + "_full_info.bin";
        read_particles_binary_into_arrays(checkpoint_file, num_particles, ids,
            radii, aspects, shaft_lengths, densities, volumes, masses, moments_of_inertia, avg_inertias,
            kns, ens, ets, fric_coefs,
            CoMs, oris, endpoints1, endpoints2, tvelos, avelos, forces, torques, accelerations, angular_accelerations,
            old_interactions, new_interactions, combis);

        std::string dump_local = load_data_location + "dump.rod";
        std::array<double,3> L{};
        get_box_size_simple(dump_local, L);
        system_dimensions.x = L[0]; system_dimensions.y = L[1]; system_dimensions.z = L[2];

        if (positions_only) {
            for (int i = 0; i < num_particles; ++i) {
                masses[i] = volumes[i] * density;
                float inertia_xx = h_PI * density * (1.0f/12.0f * powf(radii[i], 2.0f) * powf(shaft_lengths[i], 3.0f)
                                    + 8.0f/15.0f * powf(radii[i], 5.0f)
                                    + 1.0f/3.0f * powf(radii[i], 3.0f) * powf(shaft_lengths[i], 2.0f)
                                    + 3.0f/4.0f * powf(radii[i], 4.0f) * shaft_lengths[i]);
                float inertia_zz = h_PI * density * (0.5f * powf(radii[i], 4.0f) * shaft_lengths[i] + 8.0f/15.0f * powf(radii[i], 5.0f));
                moments_of_inertia[i] = make_float3(inertia_xx, inertia_xx, inertia_zz);
                avg_inertias[i] = (inertia_xx + inertia_xx + inertia_zz) / 3.0f;
                kns[i] = kn; ens[i] = en; ets[i] = et; fric_coefs[i] = friction_coef;
                tvelos[i] = make_float3(rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo, rand_m1_to_1()*max_tvelo);
                avelos[i] = make_float3(rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo, rand_m1_to_1()*max_avelo);
                forces[i] = make_float3(0,0,0); torques[i] = make_float3(0,0,0);
                accelerations[i] = make_float3(0,0,0); angular_accelerations[i] = make_float3(0,0,0);
                coord_nums[i] = 0;
            }
        }

        for (int i = 0; i < num_particles; i++) total_rods_vol += volumes[i];

        float system_volume = system_dimensions.x * system_dimensions.y * system_dimensions.z;
        float target_scale_factor = powf((total_rods_vol / vol_frac) / system_volume, 1.0f/3.0f);
        target_dimensions.x = system_dimensions.x * target_scale_factor;
        target_dimensions.y = system_dimensions.y * target_scale_factor;
        target_dimensions.z = system_dimensions.z * target_scale_factor;
    }
}


// ============================================================
// PART 4b: Metal relaxation + new_generation_process
// ============================================================

// Helper: allocate a zeroed shared MTLBuffer
static id<MTLBuffer> make_zero_buf(id<MTLDevice> dev, size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memset(b.contents, 0, bytes);
    return b;
}

// relaxation: Metal version. Pass MetalSim for PSOs/queue/device.
// b_* are the generation-local particle buffers created in new_generation_process.

// relaxation: Metal version using _ex dispatch wrappers with local buffers
static void relaxation_metal(
    MetalSim &ms,
    int num_particles, long long combis,
    id<MTLBuffer> b_CoMs,  id<MTLBuffer> b_oris,
    id<MTLBuffer> b_shafts, id<MTLBuffer> b_radii,
    id<MTLBuffer> b_masses, id<MTLBuffer> b_avg_inerts,
    id<MTLBuffer> b_coord_nums,
    id<MTLBuffer> b_kns,   id<MTLBuffer> b_ens,
    id<MTLBuffer> b_ets,   id<MTLBuffer> b_fric_coefs,
    id<MTLBuffer> b_tvels, id<MTLBuffer> b_avels,
    id<MTLBuffer> b_forces, id<MTLBuffer> b_torques,
    id<MTLBuffer> b_cn_stresses, id<MTLBuffer> b_ct_stresses, id<MTLBuffer> b_l_stresses,
    id<MTLBuffer> b_old_int, id<MTLBuffer> b_new_int,
    float dt, float fluid_viscosity,
    float lub_min_sep, float lub_max_sep,
    float ee_manual_weight, float ss_manual_weight, float es_manual_weight,
    float3 system_dimensions, int3 periodic_boundaries,
    id<MTLBuffer> b_moi,
    id<MTLBuffer> b_endpoints1, id<MTLBuffer> b_endpoints2,
    id<MTLBuffer> b_volumes,
    float fluid_density,
    id<MTLBuffer> b_taccs, id<MTLBuffer> b_aaccs,
    float relaxed_ke_thresh, int timesteps_per_check,
    bool allow_rotations, bool drag_on_free_only)
{
    int num_bins = 1;
    id<MTLBuffer> b_velo     = make_zero_buf(ms.device, (num_bins+1) * sizeof(float));
    id<MTLBuffer> b_grad     = make_zero_buf(ms.device, num_bins * sizeof(float));
    id<MTLBuffer> b_min_cont = make_zero_buf(ms.device, sizeof(float));
    id<MTLBuffer> b_min_frc  = make_zero_buf(ms.device, sizeof(float));
    id<MTLBuffer> b_min_torq = make_zero_buf(ms.device, sizeof(float));
    id<MTLBuffer> b_KE       = make_zero_buf(ms.device, num_particles * sizeof(float));
    id<MTLBuffer> b_num_ref  = make_zero_buf(ms.device, sizeof(int));
    *(int*)b_num_ref.contents = num_particles;

    float bin_size   = system_dimensions.z;
    float max_height = system_dimensions.z;

    int contact_toggle  = 1;
    int friction_toggle = 1;
    int lub_toggle      = 0;
    int drag_toggle     = 1;
    int lift_toggle     = 0;
    bool gravity        = false;
    float3 grav_dir     = {0,0,0};
    float LEBC_shift    = 0.0f;
    float LEBC_velo     = 0.0f;
    bool gen_phase      = true;

    bool relaxed = false;

    while (!relaxed) {
        for (int ts = 0; ts < timesteps_per_check; ++ts) {
            *(float*)b_min_cont.contents = 1e6f;
            *(float*)b_min_frc.contents  = 1e6f;
            *(float*)b_min_torq.contents = 1e6f;

            step_gpu_relax(ms, num_particles, combis,
                gravity, grav_dir, (bool)friction_toggle,
                dt, fluid_viscosity, lub_min_sep, lub_max_sep,
                ee_manual_weight, ss_manual_weight, es_manual_weight,
                contact_toggle, friction_toggle, lub_toggle,
                system_dimensions, periodic_boundaries,
                LEBC_shift, LEBC_velo, gen_phase,
                drag_toggle, lift_toggle,
                num_bins, bin_size, fluid_density, max_height, allow_rotations,
                b_CoMs, b_oris, b_endpoints1, b_endpoints2,
                b_tvels, b_avels, b_forces, b_torques,
                b_taccs, b_aaccs,
                b_masses, b_shafts, b_radii,
                b_kns, b_ens, b_ets, b_fric_coefs,
                b_avg_inerts, b_coord_nums,
                b_cn_stresses, b_ct_stresses, b_l_stresses,
                b_old_int, b_new_int, b_moi,
                b_num_ref,
                b_velo, b_grad,
                b_min_cont, b_min_frc, b_min_torq);
        }

        // compute avg energy
        dispatch_find_energy_kernel(ms, num_particles,
            b_masses, b_tvels, b_moi, b_avels, b_KE);
        float *ke_ptr = (float*)b_KE.contents;
        double sum = 0.0;
        for (int i = 0; i < num_particles; i++) sum += ke_ptr[i];
        float avg_energy = (float)(sum / num_particles);

        float avg_coord = find_avg_coord_num(num_particles, b_coord_nums);

        if (avg_energy <= relaxed_ke_thresh || avg_coord == 0) relaxed = true;

        std::cout << std::fixed << std::setprecision(4)
                  << "energy:  " << avg_energy << "/" << relaxed_ke_thresh << "\n";
        std::cout << std::fixed << std::setprecision(4)
                  << "coord number:  " << avg_coord << "\n";
        std::cout << "---\n";
    }
}

void new_generation_process(
    MetalSim &ms,
    int &num_particles, float radius_1, float radius_2, float prop_1, float prop_2,
    float aspect, float density, float kn, float en, float et, float friction_coef,
    float max_tvelo, float max_avelo, float3 &system_dimensions,
    int *ids, float *radii, float *aspects, float *shaft_lengths, float *densities,
    float *volumes, float *masses, float3 *moments_of_inertia, float *avg_inertias, int *coord_num,
    float *kns, float *ens, float *ets, float *fric_coefs,
    float h_max,
    float3 *CoMs, float3 *oris, float3 *endpoints1, float3 *endpoints2,
    float3 *tvelos, float3 *avelos,
    float3 *forces, float3 *torques, float3 *accelerations, float3 *angular_accelerations,
    float3 *old_interactions, float3 *new_interactions, long long &combis,
    bool safe_generation, bool set_vol_frac, bool fixed_system_size, float vol_frac,
    float oop, float3 ref_direction,
    bool load_system, std::string load_data_location, int starting_time_step,
    bool positions_only, bool continue_sim,
    float dt, float fluid_viscosity, float lub_min_sep, float lub_max_sep,
    float ee_manual_weight, float ss_manual_weight, float es_manual_weight,
    int3 periodic_boundaries, float fluid_density,
    float relaxed_ke_thresh, int timesteps_per_check, float intial_vf,
    bool allow_rotations, bool drag_on_free_only, float delta_vf_step_size,
    bool from_polyfile, std::string polyfile_loc)
{
    float3 target_dimensions{};
    float total_rods_vol = 0.0f;

    if (!from_polyfile) {
        generate_particles3(num_particles, radius_1, radius_2, prop_1, prop_2,
            aspect, density, kn, en, et, friction_coef, max_tvelo, max_avelo,
            system_dimensions, target_dimensions,
            ids, radii, aspects, shaft_lengths, densities, volumes, masses,
            moments_of_inertia, avg_inertias, coord_num,
            kns, ens, ets, fric_coefs, h_max,
            CoMs, oris, endpoints1, endpoints2, tvelos, avelos,
            forces, torques, accelerations, angular_accelerations,
            old_interactions, new_interactions, combis,
            safe_generation, set_vol_frac, fixed_system_size, vol_frac,
            oop, ref_direction,
            load_system, load_data_location, starting_time_step,
            positions_only, continue_sim,
            intial_vf, total_rods_vol);
    } else {
        generate_particles_from_polyfile(num_particles, density, kn, en, et, friction_coef,
            max_tvelo, max_avelo, system_dimensions, target_dimensions,
            ids, radii, aspects, shaft_lengths, densities, volumes, masses,
            moments_of_inertia, avg_inertias, coord_num,
            kns, ens, ets, fric_coefs, h_max,
            CoMs, oris, endpoints1, endpoints2, tvelos, avelos,
            forces, torques, accelerations, angular_accelerations,
            old_interactions, new_interactions, combis,
            vol_frac, oop, ref_direction,
            intial_vf, total_rods_vol, polyfile_loc);
    }

    // Allocate local Metal buffers for the generation phase
    id<MTLDevice> dev = ms.device;
    auto mkbuf_f3 = [&](size_t n) -> id<MTLBuffer> {
        return [dev newBufferWithLength:n*sizeof(float3) options:MTLResourceStorageModeShared];
    };
    auto mkbuf_f  = [&](size_t n) -> id<MTLBuffer> {
        return [dev newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
    };
    auto mkbuf_i  = [&](size_t n) -> id<MTLBuffer> {
        return [dev newBufferWithLength:n*sizeof(int) options:MTLResourceStorageModeShared];
    };
    auto mkbuf_m33= [&](size_t n) -> id<MTLBuffer> {
        return [dev newBufferWithLength:n*9*sizeof(float) options:MTLResourceStorageModeShared];
    };

    id<MTLBuffer> g_CoMs    = mkbuf_f3(num_particles);
    id<MTLBuffer> g_oris    = mkbuf_f3(num_particles);
    id<MTLBuffer> g_ep1     = mkbuf_f3(num_particles);
    id<MTLBuffer> g_ep2     = mkbuf_f3(num_particles);
    id<MTLBuffer> g_tvels   = mkbuf_f3(num_particles);
    id<MTLBuffer> g_avels   = mkbuf_f3(num_particles);
    id<MTLBuffer> g_forces  = mkbuf_f3(num_particles);
    id<MTLBuffer> g_torques = mkbuf_f3(num_particles);
    id<MTLBuffer> g_taccs   = mkbuf_f3(num_particles);
    id<MTLBuffer> g_aaccs   = mkbuf_f3(num_particles);
    id<MTLBuffer> g_moi     = mkbuf_f3(num_particles);
    id<MTLBuffer> g_old_int = mkbuf_f3((size_t)combis);
    id<MTLBuffer> g_new_int = mkbuf_f3((size_t)combis);

    id<MTLBuffer> g_shafts  = mkbuf_f(num_particles);
    id<MTLBuffer> g_radii   = mkbuf_f(num_particles);
    id<MTLBuffer> g_masses  = mkbuf_f(num_particles);
    id<MTLBuffer> g_vols    = mkbuf_f(num_particles);
    id<MTLBuffer> g_kns     = mkbuf_f(num_particles);
    id<MTLBuffer> g_ens     = mkbuf_f(num_particles);
    id<MTLBuffer> g_ets     = mkbuf_f(num_particles);
    id<MTLBuffer> g_frcs    = mkbuf_f(num_particles);
    id<MTLBuffer> g_ainert  = mkbuf_f(num_particles);

    id<MTLBuffer> g_coord   = mkbuf_i(num_particles);
    id<MTLBuffer> g_cn      = mkbuf_m33(num_particles);
    id<MTLBuffer> g_ct      = mkbuf_m33(num_particles);
    id<MTLBuffer> g_lub     = mkbuf_m33(num_particles);

    // Copy CPU arrays to local buffers
    memcpy(g_CoMs.contents,    CoMs,                  num_particles*sizeof(float3));
    memcpy(g_oris.contents,    oris,                  num_particles*sizeof(float3));
    memcpy(g_ep1.contents,     endpoints1,            num_particles*sizeof(float3));
    memcpy(g_ep2.contents,     endpoints2,            num_particles*sizeof(float3));
    memcpy(g_tvels.contents,   tvelos,                num_particles*sizeof(float3));
    memcpy(g_avels.contents,   avelos,                num_particles*sizeof(float3));
    memcpy(g_forces.contents,  forces,                num_particles*sizeof(float3));
    memcpy(g_torques.contents, torques,               num_particles*sizeof(float3));
    memcpy(g_taccs.contents,   accelerations,         num_particles*sizeof(float3));
    memcpy(g_aaccs.contents,   angular_accelerations, num_particles*sizeof(float3));
    memcpy(g_moi.contents,     moments_of_inertia,    num_particles*sizeof(float3));
    memcpy(g_old_int.contents, old_interactions,      combis*sizeof(float3));
    memcpy(g_new_int.contents, new_interactions,      combis*sizeof(float3));
    memcpy(g_shafts.contents,  shaft_lengths,         num_particles*sizeof(float));
    memcpy(g_radii.contents,   radii,                 num_particles*sizeof(float));
    memcpy(g_masses.contents,  masses,                num_particles*sizeof(float));
    memcpy(g_vols.contents,    volumes,               num_particles*sizeof(float));
    memcpy(g_kns.contents,     kns,                   num_particles*sizeof(float));
    memcpy(g_ens.contents,     ens,                   num_particles*sizeof(float));
    memcpy(g_ets.contents,     ets,                   num_particles*sizeof(float));
    memcpy(g_frcs.contents,    fric_coefs,            num_particles*sizeof(float));
    memcpy(g_ainert.contents,  avg_inertias,          num_particles*sizeof(float));
    memcpy(g_coord.contents,   coord_num,             num_particles*sizeof(int));
    memset(g_cn.contents,  0, num_particles*9*sizeof(float));
    memset(g_ct.contents,  0, num_particles*9*sizeof(float));
    memset(g_lub.contents, 0, num_particles*9*sizeof(float));

    // Allocate and populate pair consts buffer for use during relaxation.
    // main() will reallocate it again after new_generation_process returns.
    ms.buf_pair_consts = [dev newBufferWithLength:(size_t)combis * sizeof(PairConsts)
                                         options:MTLResourceStorageModeShared];
    precompute_pair_consts(ms,
        (const float*)g_kns.contents, (const float*)g_ens.contents,
        (const float*)g_ets.contents, (const float*)g_masses.contents,
        num_particles);

    bool atDesiredSize = false;
    bool first_loop    = true;

    while (!atDesiredSize) {
        bool allow_rotations2 = allow_rotations;
        if (first_loop) allow_rotations2 = false;

        relaxation_metal(ms, num_particles, combis,
            g_CoMs, g_oris, g_shafts, g_radii, g_masses, g_ainert, g_coord,
            g_kns, g_ens, g_ets, g_frcs, g_tvels, g_avels, g_forces, g_torques,
            g_cn, g_ct, g_lub, g_old_int, g_new_int,
            dt, fluid_viscosity, lub_min_sep, lub_max_sep,
            ee_manual_weight, ss_manual_weight, es_manual_weight,
            system_dimensions, periodic_boundaries,
            g_moi, g_ep1, g_ep2, g_vols, fluid_density,
            g_taccs, g_aaccs,
            relaxed_ke_thresh, timesteps_per_check,
            allow_rotations2, drag_on_free_only);

        first_loop = false;

        if (system_dimensions.x <= target_dimensions.x) {
            atDesiredSize = true;
            relaxation_metal(ms, num_particles, combis,
                g_CoMs, g_oris, g_shafts, g_radii, g_masses, g_ainert, g_coord,
                g_kns, g_ens, g_ets, g_frcs, g_tvels, g_avels, g_forces, g_torques,
                g_cn, g_ct, g_lub, g_old_int, g_new_int,
                dt, fluid_viscosity, lub_min_sep, lub_max_sep,
                ee_manual_weight, ss_manual_weight, es_manual_weight,
                system_dimensions, periodic_boundaries,
                g_moi, g_ep1, g_ep2, g_vols, fluid_density,
                g_taccs, g_aaccs,
                relaxed_ke_thresh, timesteps_per_check,
                true, false);
        } else {
            shrink_system(ms, system_dimensions, target_dimensions, delta_vf_step_size,
                          total_rods_vol, g_CoMs, num_particles);
        }

        float system_volume = system_dimensions.x * system_dimensions.y * system_dimensions.z;
        float curr_vf = total_rods_vol / system_volume;
        std::cout << "-------------------------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(3) << "curr vf:  " << curr_vf << "\n";
        std::cout << std::fixed << std::setprecision(3) << "targ vf:  " << vol_frac << "\n";
        std::cout << "-------------------------------------------------------------\n";
    }

    // Copy results back to CPU arrays (CoMs, oris, endpoints only, matching CUDA version)
    memcpy(CoMs,       g_CoMs.contents, num_particles*sizeof(float3));
    memcpy(oris,       g_oris.contents, num_particles*sizeof(float3));
    memcpy(endpoints1, g_ep1.contents,  num_particles*sizeof(float3));
    memcpy(endpoints2, g_ep2.contents,  num_particles*sizeof(float3));
    // ARC releases local buffers automatically
}


// ============================================================
// main()
// ============================================================

int main(int argc, char** argv) {

    Timer clock;
    clock.start();
    Helper helper;

    std::string cfgfile = "config.txt";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-in" && i+1 < argc) { cfgfile = argv[i+1]; i++; }
    }

    auto config = loadConfig(cfgfile);

    auto getStr = [&](const std::string &k, const std::string &def) -> std::string {
        std::string key = k;
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c){ return std::tolower(c); });
        auto it = config.find(key);
        return (it == config.end()) ? def : it->second;
    };

    int seed = toInt(getStr("random_seed","1"));
    srand(seed);

    int num_particles   = getStr("num_particles","1000").empty() ? 1000 : toInt(getStr("num_particles","1000"));
    float radius_1      = toFloat(getStr("radius_1","1.0"));
    float prop_1        = toFloat(getStr("prop_1","0.5"));
    float radius_2      = toFloat(getStr("radius_2","1.4"));
    float prop_2        = toFloat(getStr("prop_2","0.5"));
    float aspect        = toFloat(getStr("aspect","5.0"));

    bool from_poly_file  = toBool(getStr("from_poly_file","false"));
    std::string poly_file= getStr("poly_file","D:/sims/lubfix_v6_3_abs_90/rod_info/37000000_full_info.csv");
    bool safe_generation = toBool(getStr("safe_generation","false"));

    float volume_fraction = toFloat(getStr("volume_fraction","0.40"));
    bool set_vol_frac   = toBool(getStr("set_vol_frac","true"));
    bool fixed_sys_size = toBool(getStr("fixed_sys_size","false"));
    float oop           = toFloat(getStr("oop","0.01"));
    Float3 ref_direction_f = parseVector3f(getStr("ref_direction","[1.0,0.0,0.0]"));
    float3 ref_direction   = make_float3(ref_direction_f.x, ref_direction_f.y, ref_direction_f.z);

    bool special_generation  = toBool(getStr("special_generation","false"));
    float relaxed_ke_thresh  = toFloat(getStr("relaxed_ke_thresh","0.01"));
    int timesteps_per_check  = toInt(getStr("timesteps_per_check","1000"));
    float intial_vf          = toFloat(getStr("intial_vf","0.25"));
    bool allow_rotations     = toBool(getStr("allow_rotations","true"));
    bool drag_on_free_only   = toBool(getStr("drag_on_free_only","false"));
    float delta_vf_step_size = toFloat(getStr("delta_vf_step_size","0.01"));

    float density       = toFloat(getStr("density","1.0"));
    float kn            = toFloat(getStr("kn","1500000.0"));
    float en            = toFloat(getStr("en","0.5"));
    float et            = toFloat(getStr("et","0.5"));
    float friction_coef = toFloat(getStr("friction_coef","0.5"));
    float max_tvelo     = toFloat(getStr("max_tvelo","0.0"));
    float max_avelo     = toFloat(getStr("max_avelo","0.0"));

    Float3 sys_dim_f = parseVector3f(getStr("system_dimensions","[31.94,31.94,31.94]"));
    float3 system_dimensions = make_float3(sys_dim_f.x, sys_dim_f.y, sys_dim_f.z);

    auto pb_ints = parseIntVector(getStr("periodic_boundaries","[1,1,1]"));
    int3 periodic_boundaries = make_int3(
        (pb_ints.size()>0?pb_ints[0]:1),
        (pb_ints.size()>1?pb_ints[1]:1),
        (pb_ints.size()>2?pb_ints[2]:1));

    float fluid_viscosity = toFloat(getStr("fluid_viscosity","15.0"));
    float fluid_density   = toFloat(getStr("fluid_density","1.0"));
    float lub_min_sep     = toFloat(getStr("lub_min_sep","0.025"));
    float lub_max_sep     = toFloat(getStr("lub_max_sep","0.9"));

    float ss_manual_weight = toFloat(getStr("ss_manual_weight","0.1"));
    float es_manual_weight = toFloat(getStr("es_manual_weight","1.0"));
    float ee_manual_weight = toFloat(getStr("ee_manual_weight","5.0"));

    bool gravity      = toBool(getStr("gravity","false"));
    Float3 grav_dir_f = parseVector3f(getStr("grav_dir","[0.0,-0.0,-1.0]"));
    float3 grav_dir   = normalize(make_float3(grav_dir_f.x, grav_dir_f.y, grav_dir_f.z));

    int contact_toggle  = toInt(getStr("contact_toggle","1"));
    int friction_toggle = toInt(getStr("friction_toggle","1"));
    int lub_toggle      = toInt(getStr("lub_toggle","1"));
    int drag_toggle     = toInt(getStr("drag_toggle","1"));
    int lift_toggle     = toInt(getStr("lift_toggle","1"));

    int gpu_batch_size  = toInt(getStr("gpu_batch_size","4"));
    if (gpu_batch_size < 1) gpu_batch_size = 1;

    int fixed_stress    = toInt(getStr("fixed_stress","0"));
    float controller_gain = toFloat(getStr("controller_gain","0.00001"));

    bool set_uniform_gradient = toBool(getStr("set_uniform_gradient","true"));
    float uniform_gradient    = toFloat(getStr("uniform_gradient","0.07"));

    auto velo_values_d = parseVectorFloat(getStr("fluid_velocity_profile","[0.0,2.82]"));
    int num_velos_specified = (int)velo_values_d.size();
    if (num_velos_specified < 1) {
        velo_values_d = parseVectorFloat("[0.0,2.82]");
        num_velos_specified = (int)velo_values_d.size();
    }
    float* fluid_velocity_profile = new float[num_velos_specified];
    for (int i = 0; i < num_velos_specified; i++) fluid_velocity_profile[i] = (float)velo_values_d[i];

    auto fs_values_d = parseVectorFloat(getStr("fixed_stress_profile","[25.0]"));
    int num_fixed_stress_bins = (int)fs_values_d.size();
    float* fixed_stress_profile = new float[num_fixed_stress_bins];
    for (int i = 0; i < num_fixed_stress_bins; i++) fixed_stress_profile[i] = (float)fs_values_d[i];

    int num_bins = 1;
    if (fixed_stress)  num_bins = num_fixed_stress_bins;
    else               num_bins = std::max(1, num_velos_specified - 1);

    float* fluid_velocity_gradient_profile = new float[num_bins]();

    Float3 n_ref_f = parseVector3f(getStr("order_ref_direction","[1.0,-0.0,-0.0]"));
    float3 n_ref   = normalize(make_float3(n_ref_f.x, n_ref_f.y, n_ref_f.z));

    float dt            = toFloat(getStr("dt","0.00005"));
    bool dynamic_dt     = toBool(getStr("dynamic_dt","true"));
    float dynamic_dt_range = toFloat(getStr("dynamic_dt_range","10.0"));
    float min_dt        = dt / dynamic_dt_range;
    float max_dt        = dt * dynamic_dt_range;
    float dyn_dt_scale  = toFloat(getStr("dyn_dt_scale","1.0"));

    long long time_steps = toLongLong(getStr("time_steps","2040816320000"));
    float cut_off        = toFloat(getStr("cut_off","100.0"));

    bool doFluidOut      = toBool(getStr("doFluidOut","true"));
    bool doVisOut        = toBool(getStr("doVisOut","true"));
    bool doSimpleOut     = toBool(getStr("doSimpleOut","true"));
    bool doCheckpointOut = toBool(getStr("doCheckpointOut","true"));
    OutputFlags output_flags{doFluidOut, doVisOut, doSimpleOut, doCheckpointOut};

    bool use_strains     = toBool(getStr("use_strains","true"));
    float print_interval = toFloat(getStr("print_interval","0.01"));
    float save_interval  = toFloat(getStr("save_interval","0.01"));
    float full_info_interval = toFloat(getStr("full_info_interval","1.0"));

    std::string save_name          = getStr("save_name","test");
    std::string file_path          = getStr("file_path","Outputs/");

    bool load_system              = toBool(getStr("load_system","false"));
    std::string load_data_location= getStr("load_data_location","D:/sims/lubfix_v6_3_abs_90/rod_info/37000000_full_info.bin");
    int starting_time_step        = toInt(getStr("starting_time_step","37000000"));
    bool positions_only           = toBool(getStr("positions_only","false"));
    bool continue_sim             = toBool(getStr("continue_sim","true"));

    // --- adjust particle counts matching CUDA pre-gen logic ---
    if (set_vol_frac) {
        prop_1 = prop_1 / (prop_1 + prop_2);
        prop_2 = 1.0f - prop_1;

        int num_particle_1{}, num_particle_2{};
        float rod_vol_1 = (4.0f/3.0f * h_PI * powf(radius_1,3.0f) + h_PI * 2.0f * radius_1 * (aspect-1.0f) * radius_1 * radius_1);
        float rod_vol_2 = (4.0f/3.0f * h_PI * powf(radius_2,3.0f) + h_PI * 2.0f * radius_2 * (aspect-1.0f) * radius_2 * radius_2);
        float rel_vol_1 = rod_vol_1 * prop_1;
        float rel_vol_2 = rod_vol_2 * prop_2;

        if (fixed_sys_size) {
            float sys_volume = system_dimensions.x * system_dimensions.y * system_dimensions.z;
            float aim = volume_fraction * sys_volume;
            num_particles = int(roundf(aim/(rel_vol_1 + rel_vol_2)));
            num_particle_1 = num_particles * prop_1;
            num_particle_2 = num_particles * prop_2;
            num_particles = num_particle_1 + num_particle_2;
        } else if (safe_generation) {
            float min_dim = 4 * fmaxf(radius_1, radius_2) * (aspect + lub_max_sep);
            num_particle_1 = num_particles * prop_1;
            num_particle_2 = num_particles * prop_2;
            num_particles = num_particle_1 + num_particle_2;
            float aim = (num_particle_1 * rod_vol_1) + (num_particle_2 * rod_vol_2);
            float sv = aim / volume_fraction;
            float ov = system_dimensions.x * system_dimensions.y * system_dimensions.z;
            float sf = powf(sv/ov, 1.0f/3.0f);
            system_dimensions.x = fmaxf(system_dimensions.x*sf, min_dim);
            system_dimensions.y = fmaxf(system_dimensions.y*sf, min_dim);
            system_dimensions.z = fmaxf(system_dimensions.z*sf, min_dim);
            float sysv = system_dimensions.x * system_dimensions.y * system_dimensions.z;
            aim = volume_fraction * sysv;
            num_particles = int(roundf(aim/(rel_vol_1 + rel_vol_2)));
            num_particle_1 = num_particles * prop_1;
            num_particle_2 = num_particles * prop_2;
            num_particles = num_particle_1 + num_particle_2;
        }
    }

    long long combis{};
    float LEBC_shift = 0.0f;
    float LEBC_velo  = 0.0f;

    std::string header_loc = load_data_location + "rod_info/" + std::to_string(starting_time_step) + "_full_info.bin";
    if (load_system)    { read_particles_header(header_loc, num_particles, combis, LEBC_shift); }
    if (from_poly_file) { num_particles = numrods_polyfile(poly_file); }

    combis = upper_tri_buf_size(num_particles);

    float last_print = 0, last_save = 0, last_full_info = 0;
    bool doPrint = false, doSave = false, doFull_info = false;
    std::vector<float> batch_lebc_shifts(gpu_batch_size);

    // Metal initialisation
    std::string metallib_path = getStr("metallib_path","rods_kernels.metallib");
    {
        namespace fs = std::filesystem;
        fs::path p(metallib_path);
        if (p.is_relative()) {
            fs::path exe_dir = fs::canonical(fs::path(argv[0])).parent_path();
            metallib_path = (exe_dir / p).string();
        }
    }
    MetalSim ms;
    ms.TPB  = toInt(getStr("TPB","256"));
    ms.TPB2 = toInt(getStr("TPB2","16"));
    ms.TPB3 = toInt(getStr("TPB3","256"));
    metal_init(ms, metallib_path);

    // Allocate CPU particle arrays
    int*    ids          = new int[num_particles]();
    float*  radii        = new float[num_particles]();
    float*  aspects_arr  = new float[num_particles]();
    float*  shaft_lengths= new float[num_particles]();
    float*  densities    = new float[num_particles]();
    float*  volumes      = new float[num_particles]();
    float*  masses       = new float[num_particles]();
    float3* moments_of_inertia     = new float3[num_particles]();
    float*  avg_inertias = new float[num_particles]();
    float*  kns          = new float[num_particles]();
    float*  ens          = new float[num_particles]();
    float*  ets          = new float[num_particles]();
    float*  fric_coefs   = new float[num_particles]();
    float3* CoMs         = new float3[num_particles]();
    float3* oris         = new float3[num_particles]();
    float3* endpoints1   = new float3[num_particles]();
    float3* endpoints2   = new float3[num_particles]();
    float3* tvelos       = new float3[num_particles]();
    float3* avelos       = new float3[num_particles]();
    float3* forces       = new float3[num_particles]();
    float3* torques      = new float3[num_particles]();
    float3* accelerations           = new float3[num_particles]();
    float3* angular_accelerations   = new float3[num_particles]();
    float3* old_interactions = new float3[combis]();
    float3* new_interactions = new float3[combis]();
    int*    coord_nums   = new int[num_particles]();

    if (!special_generation) {
        generate_particles(num_particles, radius_1, radius_2, prop_1, prop_2,
            aspect, density, kn, en, et, friction_coef, max_tvelo, max_avelo, system_dimensions,
            ids, radii, aspects_arr, shaft_lengths, densities, volumes, masses,
            moments_of_inertia, avg_inertias, coord_nums,
            kns, ens, ets, fric_coefs, lub_max_sep,
            CoMs, oris, endpoints1, endpoints2, tvelos, avelos,
            forces, torques, accelerations, angular_accelerations,
            old_interactions, new_interactions, combis,
            safe_generation, set_vol_frac, fixed_sys_size, volume_fraction, oop, ref_direction,
            load_system, load_data_location, starting_time_step, positions_only, continue_sim);
    } else {
        new_generation_process(ms,
            num_particles, radius_1, radius_2, prop_1, prop_2,
            aspect, density, kn, en, et, friction_coef, max_tvelo, max_avelo, system_dimensions,
            ids, radii, aspects_arr, shaft_lengths, densities, volumes, masses,
            moments_of_inertia, avg_inertias, coord_nums,
            kns, ens, ets, fric_coefs, lub_max_sep,
            CoMs, oris, endpoints1, endpoints2, tvelos, avelos,
            forces, torques, accelerations, angular_accelerations,
            old_interactions, new_interactions, combis,
            safe_generation, set_vol_frac, fixed_sys_size, volume_fraction, oop, ref_direction,
            load_system, load_data_location, starting_time_step, positions_only, continue_sim,
            dt, fluid_viscosity, lub_min_sep, lub_max_sep,
            ee_manual_weight, ss_manual_weight, es_manual_weight,
            periodic_boundaries, fluid_density,
            relaxed_ke_thresh, timesteps_per_check, intial_vf,
            allow_rotations, drag_on_free_only, delta_vf_step_size,
            from_poly_file, poly_file);
    }

    float max_height = system_dimensions.z;
    aspect = average_property(aspects_arr, num_particles);

    if (set_uniform_gradient) {
        fixed_stress = false;
        num_bins = 1;
        num_velos_specified = num_bins + 1;
        delete[] fluid_velocity_profile;
        delete[] fluid_velocity_gradient_profile;
        fluid_velocity_profile = new float[num_velos_specified];
        fluid_velocity_gradient_profile = new float[num_bins];
        fluid_velocity_profile[0] = 0.0f;
        fluid_velocity_profile[1] = uniform_gradient * max_height;
        fluid_velocity_gradient_profile[0] = 0.0f;
    }


    float bin_size = max_height / float(num_bins);

    if (fixed_stress)  velocity_profile_eval(fluid_velocity_profile, num_bins, bin_size, fluid_velocity_gradient_profile);
    if (!fixed_stress) velocity_gradient_profile_eval(fluid_velocity_profile, num_velos_specified, bin_size, fluid_velocity_gradient_profile);

    // Allocate Metal shared buffers
    ms.num_particles     = num_particles;
    ms.combis            = combis;
    ms.num_bins          = num_bins;
    ms.num_velos_specified = num_velos_specified;
    metal_alloc_buffers(ms, num_particles, combis, num_bins, num_velos_specified);

    // Copy CPU → shared Metal buffers (memcpy on unified memory)
    memcpy(ms.buf_CoMs.contents,        CoMs,                  num_particles*sizeof(float3));
    memcpy(ms.buf_oris.contents,        oris,                  num_particles*sizeof(float3));
    memcpy(ms.buf_endpoints1.contents,  endpoints1,            num_particles*sizeof(float3));
    memcpy(ms.buf_endpoints2.contents,  endpoints2,            num_particles*sizeof(float3));
    memcpy(ms.buf_tvels.contents,       tvelos,                num_particles*sizeof(float3));
    memcpy(ms.buf_avels.contents,       avelos,                num_particles*sizeof(float3));
    memcpy(ms.buf_forces.contents,      forces,                num_particles*sizeof(float3));
    memcpy(ms.buf_torques.contents,     torques,               num_particles*sizeof(float3));
    memcpy(ms.buf_taccs.contents,       accelerations,         num_particles*sizeof(float3));
    memcpy(ms.buf_aaccs.contents,       angular_accelerations, num_particles*sizeof(float3));
    memcpy(ms.buf_moi.contents,         moments_of_inertia,    num_particles*sizeof(float3));
    memcpy(ms.buf_old_int.contents,     old_interactions,      combis*sizeof(float3));
    memcpy(ms.buf_old_int.contents,     new_interactions,      combis*sizeof(float3));  // current values go into old_int (in-place scheme)
    memcpy(ms.buf_shafts.contents,      shaft_lengths,         num_particles*sizeof(float));
    memcpy(ms.buf_radii.contents,       radii,                 num_particles*sizeof(float));
    memcpy(ms.buf_masses.contents,      masses,                num_particles*sizeof(float));
    memcpy(ms.buf_volumes.contents,     volumes,               num_particles*sizeof(float));
    memcpy(ms.buf_kns.contents,         kns,                   num_particles*sizeof(float));
    memcpy(ms.buf_ens.contents,         ens,                   num_particles*sizeof(float));
    memcpy(ms.buf_ets.contents,         ets,                   num_particles*sizeof(float));
    memcpy(ms.buf_fric_coefs.contents,  fric_coefs,            num_particles*sizeof(float));
    memcpy(ms.buf_avg_inerts.contents,  avg_inertias,          num_particles*sizeof(float));
    memcpy(ms.buf_coord_nums.contents,  coord_nums,            num_particles*sizeof(int));
    memset(ms.buf_cn_stresses.contents,    0, num_particles*9*sizeof(float));
    memset(ms.buf_ct_stresses.contents,    0, num_particles*9*sizeof(float));
    memset(ms.buf_l_stresses.contents,     0, num_particles*9*sizeof(float));
    memcpy(ms.buf_velo_profile.contents,   fluid_velocity_profile,          num_velos_specified*sizeof(float));
    memcpy(ms.buf_grad_profile.contents,   fluid_velocity_gradient_profile, num_bins*sizeof(float));
    memset(ms.buf_vol_frac_profile.contents,  0, num_bins*sizeof(float));
    memset(ms.buf_cn_stress_profile.contents, 0, num_bins*sizeof(float));
    memset(ms.buf_ct_stress_profile.contents, 0, num_bins*sizeof(float));
    memset(ms.buf_l_stress_profile.contents,  0, num_bins*sizeof(float));
    *(float*)ms.buf_min_dt_cont.contents   = 1e6f;
    *(float*)ms.buf_min_dt_force.contents  = 1e6f;
    *(float*)ms.buf_min_dt_torque.contents = 1e6f;
    *(int*)ms.buf_num_particles_ref.contents = num_particles;

    // Precompute per-pair constants (kn_eff, M_eff, en_eff, et_eff, t_c, dc_n)
    precompute_pair_consts(ms, kns, ens, ets, masses, num_particles);

    // CPU-side profile arrays
    float* vol_frac_profile    = new float[num_bins]();
    float* cn_stress_profile   = new float[num_bins]();
    float* ct_stress_profile   = new float[num_bins]();
    float* l_stress_profile    = new float[num_bins]();
    float* fluid_stress_profile= new float[num_bins]();
    float* shear_stress_profile= new float[num_bins]();

    std::string folder_path           = create_output_folder(save_name, file_path);
    std::string rod_info_folder_path  = create_output_subfolder("rod_info", folder_path);
    std::string fluid_info_folder_path= create_output_subfolder("fluid_info", folder_path);

    std::vector<std::string> order_output_header = {"time","strain","visc","cn_visc","ct_visc","lub_visc",
        "fluid_visc","S_ref","S","dir_x","dir_y","dir_z","avg_stress","avg_shearrate","avg_coord_num",
        "dt","min_dt_cont","min_dt_force","min_dt_torq"};
    createCSVFile(folder_path, "simple_output.csv", order_output_header);

    sim_settings_output(dt, time_steps, print_interval, save_interval, full_info_interval, folder_path);
    system_data_output(dt, system_dimensions, periodic_boundaries, contact_toggle, friction_toggle,
        lub_toggle, drag_toggle, lift_toggle, gravity, grav_dir, folder_path);

    double tot_strain = 0;
    double tot_time   = 0;
    int frame = 0;
    double last_print_wall_s = 0.0;
    double last_print_sim_t  = 0.0;

    // -------------------------------------------------------------------------
    // Main simulation loop
    // -------------------------------------------------------------------------
    for (long long time_step = 0; time_step < time_steps; ++time_step) {

        if (use_strains && tot_strain > cut_off)    break;
        else if (!use_strains && tot_time > cut_off) break;

        // Determine how many GPU steps to batch before the next output boundary
        {
            float ref      = use_strains ? (float)tot_strain : (float)tot_time;
            float step_ref = use_strains
                ? (float)(fluid_velocity_gradient_profile[0] * dt) : (float)dt;
            if (step_ref <= 0.0f) step_ref = (float)dt;
            float dist = std::numeric_limits<float>::max();
            if (print_interval      > 0.f) dist = std::min(dist, last_print     + print_interval      - ref);
            if (save_interval       > 0.f) dist = std::min(dist, last_save      + save_interval       - ref);
            if (full_info_interval  > 0.f) dist = std::min(dist, last_full_info + full_info_interval  - ref);
            int M_max = (dist > step_ref * 0.5f) ? (int)(dist / step_ref) : 1;
            if (M_max < 1) M_max = 1;
            int M = std::min(gpu_batch_size, M_max);
            long long remaining = time_steps - time_step;
            if ((long long)M > remaining) M = (int)remaining;
            if ((int)batch_lebc_shifts.size() < M) batch_lebc_shifts.resize(M);

            // Precompute LEBC_shift for each sub-step and update LEBC_velo
            LEBC_velo = fluid_velocity_profile[num_velos_specified-1] - fluid_velocity_profile[0];
            for (int k = 0; k < M; ++k) {
                LEBC_shift += LEBC_velo * dt;
                LEBC_shift = std::fmod(LEBC_shift, system_dimensions.x);
                if (LEBC_shift < 0.0f) LEBC_shift += system_dimensions.x;
                batch_lebc_shifts[k] = LEBC_shift;
            }

            reset_stress_profiles(num_bins,
                ms.buf_vol_frac_profile, ms.buf_cn_stress_profile,
                ms.buf_ct_stress_profile, ms.buf_l_stress_profile);
            reset_time_scales(ms.buf_min_dt_cont, ms.buf_min_dt_force, ms.buf_min_dt_torque);

            if (M <= 1) {
                step_gpu(ms,
                    gravity, grav_dir, (bool)friction_toggle,
                    dt, fluid_viscosity, lub_min_sep, lub_max_sep,
                    ee_manual_weight, ss_manual_weight, es_manual_weight,
                    contact_toggle, friction_toggle, lub_toggle,
                    system_dimensions, periodic_boundaries,
                    batch_lebc_shifts[0], LEBC_velo,
                    drag_toggle, lift_toggle,
                    num_bins, bin_size, fluid_density, max_height);
            } else {
                step_gpu_multi(ms, M, batch_lebc_shifts.data(),
                    gravity, grav_dir, (bool)friction_toggle,
                    dt, fluid_viscosity, lub_min_sep, lub_max_sep,
                    ee_manual_weight, ss_manual_weight, es_manual_weight,
                    contact_toggle, friction_toggle, lub_toggle,
                    system_dimensions, periodic_boundaries, LEBC_velo,
                    drag_toggle, lift_toggle,
                    num_bins, bin_size, fluid_density, max_height);
            }

            find_new_profiles(ms, (bool)fixed_stress,
                num_particles, system_dimensions, periodic_boundaries,
                num_bins, bin_size,
                vol_frac_profile, cn_stress_profile, ct_stress_profile, l_stress_profile,
                fluid_velocity_profile, num_velos_specified, fluid_velocity_gradient_profile,
                aspect, fluid_viscosity, fluid_stress_profile, shear_stress_profile,
                fixed_stress_profile, controller_gain);

            tot_strain += fluid_velocity_gradient_profile[0] * dt * M;
            tot_time   += dt * M;

            helper.total_strain = tot_strain;
            helper.total_time   = tot_time;
            helper.shear_rate   = fluid_velocity_gradient_profile[0];

            find_new_dt(dt, dynamic_dt,
                ms.buf_min_dt_cont, ms.buf_min_dt_force, ms.buf_min_dt_torque,
                fluid_velocity_gradient_profile, max_dt, min_dt, dyn_dt_scale, helper);

            long long steps_done = time_step + M;
            float dt_avg = (steps_done > 0) ? (float)(tot_time / (double)steps_done) : dt;

            check_print_save_checkpoint(frame, use_strains, (float)tot_strain, (float)tot_time,
                last_print, last_save, last_full_info,
                print_interval, save_interval, full_info_interval,
                (int)(time_step + M - 1), doPrint, doSave, doFull_info);

            helper.doPrint = doPrint;

            double cur_wall_s = clock.elapsed_seconds();
            double d_wall = cur_wall_s - last_print_wall_s;
            double d_sim  = tot_time   - last_print_sim_t;
            double speed  = (doPrint && d_sim > 0.0) ? d_wall / d_sim : 0.0;
            if (doPrint) { last_print_wall_s = cur_wall_s; last_print_sim_t = tot_time; }
            print_sequence(doPrint, use_strains, save_name, (float)tot_time, (float)tot_strain, cut_off,
                dt, dt_avg, clock, speed, helper);

            save_sequency(doSave, doPrint, (bool)fixed_stress, ms, num_particles, combis,
                system_dimensions, periodic_boundaries,
                num_bins, bin_size,
                vol_frac_profile, cn_stress_profile, ct_stress_profile, l_stress_profile,
                fluid_velocity_profile, num_velos_specified, fluid_velocity_gradient_profile,
                aspect, fluid_viscosity, (int)(time_step + M - 1),
                radii, shaft_lengths,
                folder_path, frame,
                (float)tot_strain, dt, fluid_density, max_height,
                fluid_info_folder_path,
                fluid_stress_profile, shear_stress_profile,
                n_ref, (float)tot_time, dt_avg, output_flags);

            checkpoint_sequence(doFull_info, ms, num_particles, combis,
                rod_info_folder_path, (int)(time_step + M - 1), LEBC_shift,
                ids, radii, aspects_arr, shaft_lengths,
                densities, volumes, masses,
                avg_inertias, kns, ens, ets, fric_coefs,
                frame, output_flags);

            // Advance outer loop counter by M-1 (for-loop increments by 1)
            time_step += (long long)(M - 1);
        }
    }

    return 0;
}

