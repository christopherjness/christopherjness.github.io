/*
 * rods_kernels.metal
 * Metal (MSL) translation of the CUDA DEM rod simulation kernels.
 * Physics is identical to the original CUDA code.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ─────────────────────────────────────────────────────────
//  CONSTANTS
// ─────────────────────────────────────────────────────────
constant float PI = 3.14159265358979323846f;

// ─────────────────────────────────────────────────────────
//  SHARED STRUCTS
// ─────────────────────────────────────────────────────────
struct mat33 { float m[9]; };

// Params structs passed via constant buffers --
// Fields match exactly what the host code writes.

struct ClearParams {
    int   num_particles;
    int   gravity;          // bool as int
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

// Precomputed constant parameters for each particle pair (i < j).
// Indexed by int_index(i, j, N).  Populated once on CPU at init.
struct PairConsts {
    float kn_eff;   // knA*knB/(knA+knB)
    float M_eff;    // massA*massB/(massA+massB)
    float en_eff;   // fmin(enA,enB)
    float et_eff;   // fmin(etA,etB)
    float t_c;      // contact time  (expensive: log+sqrt)
    float dc_n;     // normal damping coef  (-2*M_eff/t_c*log(en_eff))
    float pad0;
    float pad1;
};

// ─────────────────────────────────────────────────────────
//  FLOAT ATOMIC HELPERS
// ─────────────────────────────────────────────────────────

inline void atomic_add_f(device float* addr, float val) {
    device atomic_uint* uint_addr = (device atomic_uint*)(device void*)addr;
    uint old_u = atomic_load_explicit(uint_addr, memory_order_relaxed);
    bool success = false;
    while (!success) {
        float old_f = as_type<float>(old_u);
        uint new_u  = as_type<uint>(old_f + val);
        success = atomic_compare_exchange_weak_explicit(
            uint_addr, &old_u, new_u,
            memory_order_relaxed, memory_order_relaxed);
    }
}

inline void atomic_add_i(device int* addr, int val) {
    atomic_fetch_add_explicit((device atomic_int*)(device void*)addr,
                               val, memory_order_relaxed);
}

inline void atomic_exch_f(device float* addr, float val) {
    atomic_store_explicit((device atomic_uint*)(device void*)addr,
                          as_type<uint>(val), memory_order_relaxed);
}

inline void atomic_min_f(device float* addr, float val) {
    device atomic_uint* uint_addr = (device atomic_uint*)(device void*)addr;
    uint old_u = atomic_load_explicit(uint_addr, memory_order_relaxed);
    float old_f = as_type<float>(old_u);
    if (!(val < old_f)) return;
    bool success = false;
    while (!success) {
        uint new_u = as_type<uint>(val);
        success = atomic_compare_exchange_weak_explicit(
            uint_addr, &old_u, new_u,
            memory_order_relaxed, memory_order_relaxed);
        if (!success) {
            old_f = as_type<float>(old_u);
            if (!(val < old_f)) return;
        }
    }
}

inline void atomic_max_f(device float* addr, float val) {
    device atomic_uint* uint_addr = (device atomic_uint*)(device void*)addr;
    uint old_u = atomic_load_explicit(uint_addr, memory_order_relaxed);
    float old_f = as_type<float>(old_u);
    if (!(old_f < val)) return;
    bool success = false;
    while (!success) {
        uint new_u = as_type<uint>(val);
        success = atomic_compare_exchange_weak_explicit(
            uint_addr, &old_u, new_u,
            memory_order_relaxed, memory_order_relaxed);
        if (!success) {
            old_f = as_type<float>(old_u);
            if (!(old_f < val)) return;
        }
    }
}

// ─────────────────────────────────────────────────────────
//  MATH HELPERS  (mirrors misc_funcs.cuh device functions)
// ─────────────────────────────────────────────────────────

inline float sig_f(float x)  { return x >= 0.0f ? 1.0f : -1.0f; }
inline float mag3(float3 v)  { return length(v); }
inline float3 norm3(float3 v) {
    float l = length(v);
    return (l > 0.0f) ? v / l : float3(0.0f);
}
inline float dot3(float3 a, float3 b)  { return dot(a, b); }
inline float3 cross3(float3 a, float3 b) { return cross(a, b); }
inline float clamp_f(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

// int_index: linear index for upper triangle (i < j)
inline int int_index(int i, int j, int N) {
    return i * N - (i * (i - 1)) / 2 + (j - 1);
}

inline float3 ewmul(float3 a, float3 b) { return a * b; }

inline float3 cwdiv(float3 a, float3 b) {
    return float3(a.x/b.x, a.y/b.y, a.z/b.z);
}

// mat33 operations
inline mat33 mat33_zero() {
    mat33 r; for (int k=0;k<9;k++) r.m[k]=0.0f; return r;
}

inline mat33 outer_product(float3 a, float3 b) {
    mat33 r;
    r.m[0]=a.x*b.x; r.m[1]=a.x*b.y; r.m[2]=a.x*b.z;
    r.m[3]=a.y*b.x; r.m[4]=a.y*b.y; r.m[5]=a.y*b.z;
    r.m[6]=a.z*b.x; r.m[7]=a.z*b.y; r.m[8]=a.z*b.z;
    return r;
}

inline mat33 mat_scal_mul(mat33 A, float s) {
    mat33 r; for(int k=0;k<9;k++) r.m[k]=A.m[k]*s; return r;
}

inline mat33 elementmat_mul(mat33 A, mat33 B) {
    mat33 r; for(int k=0;k<9;k++) r.m[k]=A.m[k]*B.m[k]; return r;
}

inline mat33 elementmat_div(mat33 A, mat33 B) {
    mat33 r; for(int k=0;k<9;k++) r.m[k]=A.m[k]/B.m[k]; return r;
}

inline mat33 inverse_mat(mat33 A) {
    mat33 inv = mat33_zero();
    float det = A.m[0]*(A.m[4]*A.m[8]-A.m[5]*A.m[7])
               -A.m[1]*(A.m[3]*A.m[8]-A.m[5]*A.m[6])
               +A.m[2]*(A.m[3]*A.m[7]-A.m[4]*A.m[6]);
    if (fabs(det) < 1e-12f) return inv;
    float id = 1.0f/det;
    inv.m[0] = (A.m[4]*A.m[8]-A.m[5]*A.m[7])*id;
    inv.m[1] =-(A.m[1]*A.m[8]-A.m[2]*A.m[7])*id;
    inv.m[2] = (A.m[1]*A.m[5]-A.m[2]*A.m[4])*id;
    inv.m[3] =-(A.m[3]*A.m[8]-A.m[5]*A.m[6])*id;
    inv.m[4] = (A.m[0]*A.m[8]-A.m[2]*A.m[6])*id;
    inv.m[5] =-(A.m[0]*A.m[5]-A.m[2]*A.m[3])*id;
    inv.m[6] = (A.m[3]*A.m[7]-A.m[4]*A.m[6])*id;
    inv.m[7] =-(A.m[0]*A.m[7]-A.m[1]*A.m[6])*id;
    inv.m[8] = (A.m[0]*A.m[4]-A.m[1]*A.m[3])*id;
    return inv;
}

inline mat33 test_mat_mul(mat33 A, mat33 B) {
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

// packed_float3 ↔ float3 helpers
inline float3 ld3(packed_float3 v) { return float3(v.x, v.y, v.z); }
inline packed_float3 st3(float3 v) { return packed_float3(v.x, v.y, v.z); }

// Atomic add for all three components of a packed_float3
inline void atomic_add_f3(device packed_float3* dest, float3 v) {
    atomic_add_f((device float*)dest + 0, v.x);
    atomic_add_f((device float*)dest + 1, v.y);
    atomic_add_f((device float*)dest + 2, v.z);
}

// Atomic exchange for all three components
inline void atomic_exch_f3(device packed_float3* dest, float3 v) {
    atomic_exch_f((device float*)dest + 0, v.x);
    atomic_exch_f((device float*)dest + 1, v.y);
    atomic_exch_f((device float*)dest + 2, v.z);
}

// mat33 array: atom-add single element
// mat_arr is device float*, element i has offset i*9
inline void atomic_add_mat33_elem(device float* mat_arr, int particle_i, int elem, float val) {
    atomic_add_f(mat_arr + particle_i * 9 + elem, val);
}

// ─────────────────────────────────────────────────────────
//  INTERACTION DETECTION  (interaction_detection.cuh)
// ─────────────────────────────────────────────────────────

void interaction_check(
    float3   posA,
    float3   posB,
    float    shaft_lengthA,
    float    shaft_lengthB,
    float    radiusA,
    float    radiusB,
    float3   sys_dim,
    int3     pb,
    float    lub_max_sep,
    int      lub_toggle,
    float    LEBC_shift,
    float    LEBC_velo,
    thread float3& rel_vel_LEBC_corr,
    thread float3& r_A_eff,
    thread float3& r_B_eff,
    thread bool&   interacting)
{
    r_A_eff = posA;
    r_B_eff = posB;

    float3 r_AB = r_B_eff - r_A_eff;

    if (pb.x) r_AB.x -= sys_dim.x * round(r_AB.x / sys_dim.x);
    if (pb.y) r_AB.y -= sys_dim.y * round(r_AB.y / sys_dim.y);
    if (pb.z) {
        float n = round(r_AB.z / sys_dim.z);
        r_AB.z -= n * sys_dim.z;
        r_AB.x -= n * LEBC_shift;
        r_AB.x -= sys_dim.x * round(r_AB.x / sys_dim.x);
        r_AB.z -= sys_dim.z * round(r_AB.z / sys_dim.z);
        rel_vel_LEBC_corr.x = -n * LEBC_velo;
    }

    r_B_eff = r_A_eff + r_AB;

    float lub_cut_off_factor = lub_toggle ? lub_max_sep : 0.0f;
    float sep2 = r_AB.x*r_AB.x + r_AB.y*r_AB.y + r_AB.z*r_AB.z;
    float thr  = (shaft_lengthA/2.0f + radiusA + lub_cut_off_factor*radiusA)
               + (shaft_lengthB/2.0f + radiusB + lub_cut_off_factor*radiusB);
    if (sep2 > thr*thr) interacting = false;
}

void rod_rod_separation(
    float3   oriA,
    float3   oriB,
    float    shaft_lengthA,
    float    shaft_lengthB,
    float    radius_A,
    float    radius_B,
    thread float& separation,
    thread float3& r_A_eff,
    thread float3& r_B_eff,
    thread float3& r_c,
    thread float3& direction,
    float3   sys_dim,
    int3     pb)
{
    float3 r_AB = r_B_eff - r_A_eff;

    float dot_uAuB  = dot3(oriA, oriB);
    float dot_rABuA = dot3(r_AB, oriA);
    float dot_rABuB = dot3(r_AB, oriB);

    float denom = 1.0f - dot_uAuB * dot_uAuB;
    float lambda, psi;

    if (fabs(denom) < 1e-8f) {
        if (fabs(dot_rABuA) > 1e-8f) {
            lambda = shaft_lengthA * 0.5f * sig_f(dot_rABuA);
            psi    = lambda * dot_uAuB - dot_rABuB;
            if (fabs(psi) > shaft_lengthB * 0.5f)
                psi = shaft_lengthB * 0.5f * sig_f(psi);
        } else {
            lambda = 0.0f; psi = 0.0f;
        }
    } else {
        lambda = (dot_rABuA - dot_uAuB * dot_rABuB) / denom;
        psi    = (-dot_rABuB + dot_uAuB * dot_rABuA) / denom;
        float halfA = shaft_lengthA * 0.5f;
        float halfB = shaft_lengthB * 0.5f;
        if (fabs(lambda) > halfA || fabs(psi) > halfB) {
            float aux1 = fabs(lambda) - halfA;
            float aux2 = fabs(psi)   - halfB;
            if (aux1 > aux2) {
                lambda = halfA * sig_f(lambda);
                psi    = lambda * dot_uAuB - dot_rABuB;
                if (fabs(psi) > halfB) psi = halfB * sig_f(psi);
            } else {
                psi    = halfB * sig_f(psi);
                lambda = psi * dot_uAuB + dot_rABuA;
                if (fabs(lambda) > halfA) lambda = halfA * sig_f(lambda);
            }
        }
    }

    float3 s_A = r_A_eff + oriA * lambda;
    float3 s_B = r_B_eff + oriB * psi;
    float3 sv  = s_B - s_A;
    separation  = mag3(sv);
    direction   = norm3(sv);

    r_c = float3(s_A.x + (separation + radius_A - radius_B)/2.0f * direction.x,
                 s_A.y + (separation + radius_A - radius_B)/2.0f * direction.y,
                 s_A.z + (separation + radius_A - radius_B)/2.0f * direction.z);
}

// ─────────────────────────────────────────────────────────
//  CONTACT FORCES  (contact_force.cuh)
// ─────────────────────────────────────────────────────────

inline float3 spring_force_n_calc(float kn_eff, float delta_n, float3 dir) {
    return dir * delta_n * kn_eff;
}

inline float3 rel_vel_calc(float3 tvelA, float3 tvelB,
                            float3 avelA, float3 avelB,
                            float3 r_A_eff, float3 r_B_eff, float3 r_c) {
    return tvelB - tvelA + cross3(avelB, r_c-r_B_eff) - cross3(avelA, r_c-r_A_eff);
}

inline float spring_constant_t_calc(float M_eff, float t_c, float et_eff,
                                     float3 r_A_eff, float3 r_B_eff, float3 r_c,
                                     float avg_inert_A, float avg_inert_B) {
    float d1 = mag3(r_c - r_A_eff);
    float d2 = mag3(r_c - r_B_eff);
    float denom = 1.0f/M_eff + d1*d1/avg_inert_A + d2*d2/avg_inert_B;
    float log_et = log(et_eff);
    return (PI*PI + log_et*log_et) / (t_c * t_c * denom);
}

inline float damping_coef_t_calc(float M_eff, float t_c, float et_eff,
                                  float3 r_A_eff, float3 r_B_eff, float3 r_c,
                                  float avg_inert_A, float avg_inert_B) {
    float d1 = mag3(r_c - r_A_eff);
    float d2 = mag3(r_c - r_B_eff);
    float denom = 1.0f/M_eff + d1*d1/avg_inert_A + d2*d2/avg_inert_B;
    return -2.0f * log(et_eff) / (t_c * denom);
}

void rod_rod_collision_force(
    float  radiusA, float  radiusB,
    float  sep,
    float3 r_A_eff, float3 r_B_eff, float3 r_c,
    float  kn_eff, float M_eff, float en_eff, float et_eff, float t_c, float dc_n,
    float  avg_inert_A, float avg_inert_B,
    device int* coord_num_A, device int* coord_num_B,
    float  fric_coef_A, float fric_coef_B,
    float3 tvelA, float3 tvelB,
    float3 avelA, float3 avelB,
    float3 dir,
    device packed_float3* forceA, device packed_float3* forceB,
    device packed_float3* torqueA, device packed_float3* torqueB,
    device float* cn_stressA, device float* cn_stressB,  // mat33 flat, 9 floats each particle
    int    iA, int iB,                                    // particle indices into stress arrays
    device float* ct_stressA, device float* ct_stressB,
    float3 old_interaction,
    device packed_float3* new_interaction,
    float  dt,
    int    contact_toggle, int friction_toggle,
    float3 rel_vel_LEBC_corr,
    device float* min_dt_cont,
    int    gen_phase)
{
    if (!contact_toggle) return;

    float delta_n = (radiusA + radiusB) - sep;
    if (delta_n < 0.0f) { *new_interaction = packed_float3(0.0f); return; }

    if (delta_n > 0.0f) {
        atomic_add_i(coord_num_A, 1);
        atomic_add_i(coord_num_B, 1);
    }

    float3 directions = dir * (-1.0f);
    float3 r_Ac = r_c - r_A_eff;
    float3 r_Bc = r_c - r_B_eff;

    if (isfinite(t_c) && t_c > 0.0f) atomic_min_f(min_dt_cont, t_c);

    float3 rel_vel   = rel_vel_calc(tvelA, tvelB, avelA, avelB, r_A_eff, r_B_eff, r_c);
    rel_vel += rel_vel_LEBC_corr;
    float  rel_vel_n = dot3(rel_vel, dir);

    float3 sfn_A = spring_force_n_calc(kn_eff, delta_n, directions);
    float3 dfn_A = float3(0.0f);
    if (!gen_phase)
        dfn_A = dir * (rel_vel_n * dc_n);

    float3 Nf_A = sfn_A + dfn_A;
    float3 Nf_B = Nf_A * (-1.0f);

    float3 Nt_A = cross3(r_Ac, Nf_A);
    float3 Nt_B = cross3(r_Bc, Nf_B);

    mat33 Sn_A = outer_product(Nf_A, r_Ac);
    mat33 Sn_B = outer_product(Nf_B, r_Bc);

    atomic_add_f3(forceA,  Nf_A);
    atomic_add_f3(forceB,  Nf_B);
    atomic_add_f3(torqueA, Nt_A);
    atomic_add_f3(torqueB, Nt_B);

    atomic_add_f(cn_stressA + iA*9 + 2, Sn_A.m[2]);
    atomic_add_f(cn_stressA + iA*9 + 6, Sn_A.m[6]);
    atomic_add_f(cn_stressB + iB*9 + 2, Sn_B.m[2]);
    atomic_add_f(cn_stressB + iB*9 + 6, Sn_B.m[6]);

    if (!friction_toggle) { *new_interaction = packed_float3(0.0f); return; }

    float3 rv_t   = rel_vel - dir * dot3(rel_vel, dir);
    float rv_t_m  = mag3(rv_t);
    float  mu_eff = (fric_coef_A + fric_coef_B) * 0.5f;

    // Spring tangential force
    float kt      = spring_constant_t_calc(M_eff, t_c, et_eff,
                                            r_A_eff, r_B_eff, r_c, avg_inert_A, avg_inert_B);
    float3 incr   = rv_t * dt;
    float3 cur_dt = old_interaction + incr;
    // Direct store — only one thread ever writes this pair's slot
    *new_interaction = cur_dt;

    float  cur_dt_m = mag3(cur_dt);
    float3 t_dir    = (rv_t_m   > 1e-8f) ? norm3(rv_t)   :
                      (cur_dt_m > 1e-8f) ? norm3(cur_dt) : float3(0.0f);

    float3 Tsf_A = t_dir * cur_dt_m * kt;

    float3 Tdf_A = float3(0.0f);
    if (!gen_phase) {
        float dc_t = damping_coef_t_calc(M_eff, t_c, et_eff,
                                          r_A_eff, r_B_eff, r_c, avg_inert_A, avg_inert_B);
        Tdf_A = rv_t * dc_t;
    }

    float3 tang_A     = Tsf_A + Tdf_A;
    float3 fric_A     = t_dir * mu_eff * mag3(Nf_A);

    float3 Tf_A = (mag3(tang_A) <= mu_eff * mag3(Nf_A)) ? tang_A : fric_A;

    float3 Tf_B = Tf_A * (-1.0f);
    float3 Tt_A = cross3(r_Ac, Tf_A);
    float3 Tt_B = cross3(r_Bc, Tf_B);

    mat33 St_A = outer_product(Tf_A, r_Ac);
    mat33 St_B = outer_product(Tf_B, r_Bc);

    atomic_add_f3(forceA,  Tf_A);
    atomic_add_f3(forceB,  Tf_B);
    atomic_add_f3(torqueA, Tt_A);
    atomic_add_f3(torqueB, Tt_B);

    atomic_add_f(ct_stressA + iA*9 + 2, St_A.m[2]);
    atomic_add_f(ct_stressA + iA*9 + 6, St_A.m[6]);
    atomic_add_f(ct_stressB + iB*9 + 2, St_B.m[2]);
    atomic_add_f(ct_stressB + iB*9 + 6, St_B.m[6]);
}

// ─────────────────────────────────────────────────────────
//  LUBRICATION FORCES  (lubrication.cuh)
// ─────────────────────────────────────────────────────────

void lubrication_forces(
    float  radiusA, float  radiusB,
    float  sep,
    float3 r_A_eff, float3 r_B_eff, float3 r_c,
    float  shaft_length_A, float shaft_length_B,
    float3 oriA, float3 oriB,
    float3 tvelA, float3 tvelB,
    float3 avelA, float3 avelB,
    float3 dir,
    float3 rel_vel_LEBC_corr,
    float  viscosity,
    float  min_sep,  float  max_sep,
    float  ee_manual_weight, float ss_manual_weight, float es_manual_weight,
    int    lub_toggle,
    device packed_float3* forceA, device packed_float3* forceB,
    device packed_float3* torqueA, device packed_float3* torqueB,
    device float* l_stressA, device float* l_stressB,
    int    iA, int iB,
    float3 mom_in_A, float3 mom_in_B,
    float  massA, float massB,
    device float* min_dt_force,
    device float* min_dt_torque)
{
    if (!lub_toggle) return;
    if (sep - (radiusA + radiusB) > max_sep * (radiusA + radiusB)) return;

    float3 directions = dir * (-1.0f);

    // END-END separation
    float3 rae1 = r_A_eff + oriA * (shaft_length_A / 2.0f);
    float3 rae2 = r_A_eff - oriA * (shaft_length_A / 2.0f);
    float3 rbe1 = r_B_eff + oriB * (shaft_length_B / 2.0f);
    float3 rbe2 = r_B_eff - oriB * (shaft_length_B / 2.0f);

    float min_ee_sep = fmin(fmin(mag3(rae1-rbe1), mag3(rae1-rbe2)),
                            fmin(mag3(rae2-rbe1), mag3(rae2-rbe2)));
    float ee_weight  = fmax(1e-8f, min_ee_sep);
    float ee_sep     = fmax(min_sep*(radiusA+radiusB), min_ee_sep - (radiusA+radiusB));

    // SHAFT-SHAFT separation
    float3 r_AB          = r_B_eff - r_A_eff;
    float  dot_uAuB      = dot3(oriA, oriB);
    float  dot_rABuA     = dot3(r_AB, oriA);
    float  dot_rABuB     = dot3(r_AB, oriB);
    float  eff_half_A    = shaft_length_A / 2.0f - 2.0f * radiusA;
    float  eff_half_B    = shaft_length_B / 2.0f - 2.0f * radiusB;

    float lambda_s, psi_s;
    if (fabs(1.0f - dot_uAuB*dot_uAuB) < 1e-8f) {
        if (fabs(dot_rABuA) > 1e-8f) {
            lambda_s = eff_half_A * sig_f(dot_rABuA);
            psi_s    = lambda_s * dot_uAuB - dot_rABuB;
            if (fabs(psi_s) > shaft_length_B/2.0f)
                psi_s = eff_half_B * sig_f(psi_s);
        } else { lambda_s = 0.0f; psi_s = 0.0f; }
    } else {
        lambda_s = (dot_rABuA - dot_uAuB*dot_rABuB) / (1.0f - dot_uAuB*dot_uAuB);
        psi_s    = (-dot_rABuB + dot_uAuB*dot_rABuA) / (1.0f - dot_uAuB*dot_uAuB);
        if (fabs(lambda_s) > eff_half_A || fabs(psi_s) > eff_half_B) {
            float A1 = fabs(lambda_s) - eff_half_A;
            float A2 = fabs(psi_s)   - eff_half_B;
            if (A1 > A2) {
                lambda_s = eff_half_A * sig_f(lambda_s);
                psi_s    = lambda_s * dot_uAuB - dot_rABuB;
                if (fabs(psi_s) > eff_half_B) psi_s = eff_half_B * sig_f(psi_s);
            } else {
                psi_s    = eff_half_B * sig_f(psi_s);
                lambda_s = psi_s * dot_uAuB + dot_rABuA;
                if (fabs(lambda_s) > eff_half_A) lambda_s = eff_half_A * sig_f(lambda_s);
            }
        }
    }
    float3 s_A = r_A_eff + oriA * lambda_s;
    float3 s_B = r_B_eff + oriB * psi_s;
    float  ss_sep_  = mag3(s_B - s_A);
    float  ss_weight = fmax(1e-8f, ss_sep_ - (radiusA+radiusB));
    float  ss_sep    = fmax(min_sep*(radiusA+radiusB), ss_sep_ - (radiusA+radiusB));

    // END-SHAFT separation
    float lrae1rbs  = clamp_f(dot3((rae1-r_B_eff), oriB)/dot3(oriB,oriB), -eff_half_B, eff_half_B);
    float rae1rbs_w = mag3((r_B_eff + oriB*lrae1rbs) - rae1) - radiusB;
    float lrae2rbs  = clamp_f(dot3((rae2-r_B_eff), oriB)/dot3(oriB,oriB), -eff_half_B, eff_half_B);
    float rae2rbs_w = mag3((r_B_eff + oriB*lrae2rbs) - rae2) - radiusB;
    float lrasrbe1  = clamp_f(dot3((rbe1-r_A_eff), oriA)/dot3(oriA,oriA), -eff_half_A, eff_half_A);
    float rasrbe1_w = mag3((r_A_eff + oriA*lrasrbe1) - rbe1) - radiusA;
    float lrasrbe2  = clamp_f(dot3((rbe2-r_A_eff), oriA)/dot3(oriA,oriA), -eff_half_A, eff_half_A);
    float rasrbe2_w = mag3((r_A_eff + oriA*lrasrbe2) - rbe2) - radiusA;

    float min_es_w   = fmin(fmin(rae1rbs_w, rae2rbs_w), fmin(rasrbe1_w, rasrbe2_w));
    float es_weight  = fmax(1e-8f, min_es_w);
    float min_es_sep = fmin(fmin(rae1rbs_w-radiusA, rae2rbs_w-radiusA),
                            fmin(rasrbe1_w-radiusB,  rasrbe2_w-radiusB));
    float es_sep     = fmax(min_sep*(radiusA+radiusB), min_es_sep);

    // Force calculation
    float3 rel_vel   = tvelB - tvelA + cross3(avelB, r_c-r_B_eff) - cross3(avelA, r_c-r_A_eff);
    rel_vel         += rel_vel_LEBC_corr;
    float  rv_n      = dot3(rel_vel, directions);

    float tlen_A     = shaft_length_A + 2.0f * radiusA;
    float tlen_B     = shaft_length_B + 2.0f * radiusB;
    float avg_tlen   = (tlen_A + tlen_B) * 0.5f;
    float alt_asp_A  = tlen_A / (2.0f * radiusA);
    float alt_asp_B  = tlen_B / (2.0f * radiusB);
    float avg_asp    = (alt_asp_A + alt_asp_B) * 0.5f;
    float avg_radius = (radiusA + radiusB) * 0.5f;

    float cross_uAuB = mag3(cross3(oriA, oriB));
    float denom_ss   = (2.0f*avg_asp+1.0f)
                       * sqrt((avg_asp*avg_asp+0.25f)*cross_uAuB*cross_uAuB
                              + avg_asp*(1.0f + dot3(oriA,oriB)*dot3(oriA,oriB)));

    float3 ee_force  = directions * (viscosity * avg_tlen*avg_tlen/4.0f * 3.0f*PI * rv_n
                                     / (2.0f * avg_asp*avg_asp * ee_sep));
    float3 ss_force  = directions * (2.0f * viscosity * avg_tlen*avg_tlen/4.0f * 12.0f*PI * rv_n
                                     / (denom_ss * ss_sep));
    float3 es_force  = directions * (viscosity * avg_tlen*avg_tlen/4.0f * 4.0f*PI * rv_n
                                     / (sqrt(2.0f) * avg_asp*avg_asp * es_sep));

    float denom_w = 1.0f/(ee_manual_weight*ee_weight)
                  + 1.0f/(ss_manual_weight*ss_weight)
                  + 1.0f/(es_manual_weight*es_weight);
    float ee_inf = (1.0f/(ee_manual_weight*ee_weight)) / denom_w;
    float ss_inf = (1.0f/(ss_manual_weight*ss_weight)) / denom_w;
    float es_inf = (1.0f/(es_manual_weight*es_weight)) / denom_w;

    float3 lub_f_A = ee_force*ee_inf + ss_force*ss_inf + es_force*es_inf;
    float3 lub_f_B = lub_f_A * (-1.0f);

    float3 r_Ac = r_c - r_A_eff;
    float3 r_Bc = r_c - r_B_eff;

    float3 lub_t_A = cross3(r_Ac, lub_f_A);
    float3 lub_t_B = cross3(r_Bc, lub_f_B);

    mat33 Sl_A = outer_product(lub_f_A, r_Ac);
    mat33 Sl_B = outer_product(lub_f_B, r_Bc);

    atomic_add_f3(forceA,  lub_f_A);
    atomic_add_f3(forceB,  lub_f_B);
    atomic_add_f3(torqueA, lub_t_A);
    atomic_add_f3(torqueB, lub_t_B);

    atomic_add_f(l_stressA + iA*9 + 2, Sl_A.m[2]);
    atomic_add_f(l_stressA + iA*9 + 6, Sl_A.m[6]);
    atomic_add_f(l_stressB + iB*9 + 2, Sl_B.m[2]);
    atomic_add_f(l_stressB + iB*9 + 6, Sl_B.m[6]);

    // Dynamic dt hints
    float simp_pf  = 24.0f * viscosity * PI * avg_radius*avg_radius
                   / fmax(sep - (radiusA+radiusB), min_sep*(radiusA+radiusB))
                   * avg_asp*avg_asp / ((2.0f*avg_asp+1.0f)
                       * sqrt((avg_asp*avg_asp+0.25f)*cross_uAuB*cross_uAuB
                              + avg_asp*(1.0f+dot3(oriA,oriB)*dot3(oriA,oriB))))
                   * fabs(rv_n);
    float avg_mass    = (massA + massB) * 0.5f;
    float avg_mom_z   = (mom_in_A.z + mom_in_B.z) * 0.5f;
    float simp_force  = avg_mass / (simp_pf + 1e-30f) * avg_radius;
    float simp_torque = avg_mom_z / (0.5f * simp_pf * avg_tlen + 1e-30f);

    if (isfinite(simp_force) && simp_force > 0.0f)  atomic_min_f(min_dt_force, simp_force);
    if (isfinite(simp_torque) && simp_torque > 0.0f) atomic_min_f(min_dt_torque, simp_torque);
}

// ─────────────────────────────────────────────────────────
//  FLUID FORCES  (fluid_forces.cuh)
// ─────────────────────────────────────────────────────────

void velocity_gradient_profile_eval_d(
    constant float* velocity_profile, int num_velos,
    float bin_size, device float* gradient_profile)
{
    if (num_velos < 2) { gradient_profile[0] = 0.0f; return; }
    for (int i = 0; i < num_velos-1; ++i)
        gradient_profile[i] = (velocity_profile[i+1] - velocity_profile[i]) / bin_size;
}

float3 velocity_eval(float3 CoM,
    constant float* velocity_profile,
    constant float* gradient_profile,
    int num_grad_bins, float bin_size)
{
    int b = (int)floor(CoM.z / bin_size);
    b = clamp(b, 0, num_grad_bins-1);
    float grad = gradient_profile[b];
    float velo = velocity_profile[b+1] - (grad * (bin_size + bin_size*b - CoM.z));
    return float3(velo, 0.0f, 0.0f);
}

mat33 velocity_gradient_eval(float3 CoM,
    constant float* gradient_profile,
    int num_grad_bins, float bin_size)
{
    int b = (int)floor(CoM.z / bin_size);
    b = clamp(b, 0, num_grad_bins-1);
    float g = gradient_profile[b];
    mat33 gm = mat33_zero();
    gm.m[2] = g;
    return gm;
}

mat33 particle_velocity_gradient_eval(
    float3 CoM, float3 ep1, float3 ep2,
    float radius, float shaft_length,
    constant float* gradient_profile,
    int num_grad_bins, float bin_size)
{
    mat33 gg = velocity_gradient_eval(CoM, gradient_profile, num_grad_bins, bin_size);

    float3 dxyz;
    dxyz.x = fabs(ep1.x - ep2.x) + 2.0f * radius;
    dxyz.y = fabs(ep1.y - ep2.y) + 2.0f * radius;
    dxyz.z = fabs(ep1.z - ep2.z) + 2.0f * radius;

    mat33 dm; dm.m[0]=dxyz.x; dm.m[1]=dxyz.y; dm.m[2]=dxyz.z;
    dm.m[3]=dxyz.x; dm.m[4]=dxyz.y; dm.m[5]=dxyz.z;
    dm.m[6]=dxyz.x; dm.m[7]=dxyz.y; dm.m[8]=dxyz.z;

    mat33 dv = elementmat_mul(dm, gg);

    float3 dp;
    dp.x = 2.0f * radius;
    dp.y = 2.0f * radius;
    dp.z = 2.0f * radius + shaft_length;

    mat33 dpm; dpm.m[0]=dp.x; dpm.m[1]=dp.y; dpm.m[2]=dp.z;
    dpm.m[3]=dp.x; dpm.m[4]=dp.y; dpm.m[5]=dp.z;
    dpm.m[6]=dp.x; dpm.m[7]=dp.y; dpm.m[8]=dp.z;

    return elementmat_div(dv, dpm);
}

mat33 trans_matr_calc(float3 ori) {
    float phi   = atan2(ori.y, ori.x);
    float theta = acos(-ori.z);
    mat33 t;
    t.m[0] =  cos(phi);        t.m[1] = sin(phi);       t.m[2] = 0.0f;
    t.m[3] = -cos(theta)*sin(phi); t.m[4] = cos(theta)*cos(phi); t.m[5] = sin(theta);
    t.m[6] =  sin(theta)*sin(phi); t.m[7] =-sin(theta)*cos(phi); t.m[8] = cos(theta);
    return t;
}

mat33 resistance_particle_calc(float radius, float shaft_length) {
    float beta   = (shaft_length + 2.0f*radius) / radius;
    float factor = beta*beta - 1.0f;
    float sqf    = sqrt(factor);
    float lterm  = log(beta + sqf) / sqf;
    float k_xx   = 16.0f*factor / ((2.0f*beta*beta-3.0f)*lterm + beta);
    float k_zz   = 8.0f*factor  / ((2.0f*beta*beta-1.0f)*lterm - beta);
    mat33 k = mat33_zero();
    k.m[0] = k_xx; k.m[4] = k_xx; k.m[8] = k_zz;
    return k;
}

mat33 resistance_global_calc(float3 ori, float radius, float shaft_length) {
    mat33 T     = trans_matr_calc(ori);
    mat33 k_p   = resistance_particle_calc(radius, shaft_length);
    mat33 invT  = inverse_mat(T);
    mat33 tmp   = test_mat_mul(invT, k_p);
    return test_mat_mul(tmp, T);
}

float3 drag_force_calc(float3 CoM, float3 tvelo, float3 ori,
    float radius, float shaft_length,
    constant float* velocity_profile, constant float* gradient_profile,
    int num_grad_bins, float bin_size, float viscosity)
{
    float3 fv  = velocity_eval(CoM, velocity_profile, gradient_profile, num_grad_bins, bin_size);
    mat33  kg  = resistance_global_calc(ori, radius, shaft_length);
    float3 kd  = float3(kg.m[0], kg.m[4], kg.m[8]);
    return ewmul(kd, (fv - tvelo)) * (viscosity * PI * radius);
}

float3 lift_force_calc(float3 CoM, float3 tvelo, float3 ori,
    float radius, float shaft_length,
    constant float* velocity_profile, constant float* gradient_profile,
    int num_grad_bins, float bin_size, float viscosity, float fluid_density)
{
    mat33 lc = mat33_zero();
    lc.m[0] = 0.0501f; lc.m[1] = 0.0329f;
    lc.m[3] = 0.0182f; lc.m[4] = 0.0173f;
    lc.m[8] = 0.0373f;

    mat33  kg   = resistance_global_calc(ori, radius, shaft_length);
    float3 fv   = velocity_eval(CoM, velocity_profile, gradient_profile, num_grad_bins, bin_size);
    mat33  vg   = velocity_gradient_eval(CoM, gradient_profile, num_grad_bins, bin_size);

    float cp = PI*PI * viscosity * radius*radius / sqrt(viscosity / fluid_density);

    mat33 gp = mat33_zero();
    for (int k=0; k<9; k++)
        if (vg.m[k] != 0.0f)
            gp.m[k] = vg.m[k] / sqrt(fabs(vg.m[k]));

    mat33 klk = elementmat_mul(elementmat_mul(kg, lc), kg);

    float3 vd = fv - tvelo;
    mat33 vdm = mat33_zero();
    vdm.m[0]=vd.x; vdm.m[1]=vd.x; vdm.m[2]=vd.x;
    vdm.m[3]=vd.y; vdm.m[4]=vd.y; vdm.m[5]=vd.y;
    vdm.m[6]=vd.z; vdm.m[7]=vd.z; vdm.m[8]=vd.z;

    mat33 lm = mat_scal_mul(elementmat_mul(elementmat_mul(klk, gp), vdm), cp);

    float3 lf;
    lf.x = lm.m[0] + lm.m[3] + lm.m[6];
    lf.y = lm.m[1] + lm.m[4] + lm.m[7];
    lf.z = lm.m[2] + lm.m[5] + lm.m[8];
    return lf;
}

float3 h_torque_calc(float3 CoM, float3 avelo, float radius, float shaft_length,
    float3 ep1, float3 ep2,
    constant float* gradient_profile, int num_grad_bins, float bin_size, float viscosity)
{
    mat33 vgp = particle_velocity_gradient_eval(CoM, ep1, ep2, radius, shaft_length,
                                                 gradient_profile, num_grad_bins, bin_size);
    float beta   = (shaft_length + 2.0f*radius) / radius;
    float b2m1   = beta*beta - 1.0f;
    float sqb2m1 = sqrt(b2m1);
    float lterm  = log((beta - sqb2m1) / (beta + sqb2m1));
    float b2m1_1p5 = b2m1 * sqb2m1;  // pow(b2m1, 1.5f) = b2m1 * sqrt(b2m1)
    float alpha0 = beta*beta/b2m1 + beta/(2.0f*b2m1_1p5) * lterm;
    float beta0  = alpha0;
    float gamma0 = -2.0f/b2m1 - beta/b2m1_1p5 * lterm;

    float d_zy = 0.5f*(vgp.m[7] + vgp.m[5]);
    float d_xz = 0.5f*(vgp.m[2] + vgp.m[6]);
    float w_zy = 0.5f*(vgp.m[7] - vgp.m[5]);
    float w_xz = 0.5f*(vgp.m[2] - vgp.m[6]);
    float w_yx = 0.5f*(vgp.m[3] - vgp.m[1]);

    float3 torq;
    torq.x = 16.0f*PI*viscosity*(radius*radius*radius)*beta
           / (3.0f*(beta0 + beta*beta*gamma0))
           * ((1.0f - beta*beta)*d_zy + (1.0f + beta*beta)*(w_zy - avelo.x));
    torq.y = 16.0f*PI*viscosity*(radius*radius*radius)*beta
           / (3.0f*(alpha0 + beta*beta*gamma0))
           * ((beta*beta - 1.0f)*d_xz + (1.0f + beta*beta)*(w_xz - avelo.y));
    torq.z = 32.0f*PI*viscosity*(radius*radius*radius)*beta
           / (3.0f*(alpha0 + beta0))
           * (w_yx - avelo.z);
    return torq;
}

// ─────────────────────────────────────────────────────────
//  INTEGRATION  (integration.cuh)
// ─────────────────────────────────────────────────────────

void update_position_msl(
    float3 force, thread float3& CoM, thread float3& tvelo, thread float3& tacc,
    float mass, float LEBC_shift, float LEBC_velo, float dt,
    float3 sys, int3 pb)
{
    float3 prev_acc = tacc;
    tacc = force / mass;
    tvelo += (prev_acc * (0.5f*dt) + tacc * (0.5f*dt));

    float3 vel_half = tvelo + tacc * (0.5f*dt);
    CoM += vel_half * dt;

    if (pb.z) {
        if (CoM.z > sys.z) {
            tvelo.x += (-1.0f) * LEBC_velo;
            CoM.x   += (-1.0f) * LEBC_shift;
            CoM.z   += -sys.z;
        } else if (CoM.z < 0.0f) {
            tvelo.x += LEBC_velo;
            CoM.x   += LEBC_shift;
            CoM.z   += sys.z;
        }
    }
    if (pb.x) {
        CoM.x = fmod(CoM.x, sys.x);
        if (CoM.x < 0.0f) CoM.x += sys.x;
    }
    if (pb.y) {
        CoM.y = fmod(CoM.y, sys.y);
        if (CoM.y < 0.0f) CoM.y += sys.y;
    }
}

void update_orientation_msl(
    float3 torque, thread float3& ori,
    thread float3& ep1, thread float3& ep2,
    float3 CoM, float shaft_length,
    thread float3& avelo, thread float3& aacc,
    float3 moi, float dt, int allowRotation)
{
    if (!allowRotation) return;
    float3 prev_acc = aacc;
    aacc  = cwdiv(torque, moi);
    float3 dav = (prev_acc * (0.5f*dt) + aacc * (0.5f*dt));
    avelo += dav;

    float3 vel_half = avelo + aacc * (0.5f*dt);
    ori += cross3(vel_half, ori) * dt;
    ori  = norm3(ori);

    ep1 = CoM + ori * (shaft_length / 2.0f);
    ep2 = CoM - ori * (shaft_length / 2.0f);
}

// ─────────────────────────────────────────────────────────
//  PROFILE HELPERS  (misc_funcs.cuh device functions)
// ─────────────────────────────────────────────────────────

void add_volume_frac_contribution(
    float3 CoM, float3 ep1, float3 ep2,
    float volume, int num_bins, float bin_size,
    device float* vol_frac_profile,
    float3 sys, int3 pb)
{
    if (!isfinite(volume) || !isfinite(ep1.z) || !isfinite(ep2.z)
        || !isfinite(sys.x) || !isfinite(sys.y) || !isfinite(bin_size)
        || num_bins <= 0 || vol_frac_profile == nullptr) return;
    if (bin_size <= 0.0f) return;

    float high = fmax(ep1.z, ep2.z);
    float low  = fmin(ep1.z, ep2.z);
    float dz   = high - low;
    float bvol = sys.x * sys.y * bin_size;
    if (!(bvol > 0.0f)) return;

    const float EPS = 1e-12f;
    if (dz <= EPS) {
        int b = (int)floor(low / bin_size);
        if (b < 0 || b >= num_bins) return;
        float c = volume / bvol;
        if (isfinite(c)) atomic_add_f(&vol_frac_profile[b], c);
        return;
    }

    int start_b, end_b;
    if (pb.z) {
        start_b = (int)floor(low / bin_size);
        end_b   = (int)floor((high - 1e-6f) / bin_size);
    } else {
        start_b = max(0, (int)floor(low / bin_size));
        end_b   = min(num_bins-1, (int)floor((high - 1e-6f) / bin_size));
    }

    for (int b = start_b; b <= end_b; ++b) {
        float lo_  = b * bin_size;
        float hi_  = lo_ + bin_size;
        float lo   = fmax(lo_, low);
        float hi   = fmin(hi_, high);
        float over = hi - lo;
        if (over <= 0.0f) continue;
        float cvol = (over / dz) * volume;
        float c    = cvol / bvol;
        int rb = b;
        if (b < 0)         rb += num_bins;
        else if (b >= num_bins) rb -= num_bins;
        if (isfinite(c)) atomic_add_f(&vol_frac_profile[rb], c);
    }
}

void add_stress_contributions(
    float3 CoM, float num_bins_f, float bin_size,
    float3 cn_m26, float3 ct_m26, float3 l_m26,  // m[2] and m[6] packed as x,y,z
    device float* cn_profile, device float* ct_profile, device float* l_profile,
    float3 sys)
{
    int b    = (int)ceil(CoM.z / bin_size) - 1;
    float bv = sys.x * sys.y * bin_size;
    int nb   = (int)num_bins_f;
    if (b < 0 || b >= nb || bv <= 0.0f) return;

    float cn_c = ((cn_m26.x + cn_m26.y) * 0.5f) / bv;
    float ct_c = ((ct_m26.x + ct_m26.y) * 0.5f) / bv;
    float l_c  = ((l_m26.x  + l_m26.y)  * 0.5f) / bv;

    atomic_add_f(&cn_profile[b], cn_c);
    atomic_add_f(&ct_profile[b], ct_c);
    atomic_add_f(&l_profile[b],  l_c);
}

// ─────────────────────────────────────────────────────────
//  KERNEL: clear_for_new_time_step
// ─────────────────────────────────────────────────────────
kernel void clear_for_new_time_step(
    constant ClearParams&    p          [[buffer(0)]],
    device   packed_float3*  forces     [[buffer(1)]],
    device   packed_float3*  torques    [[buffer(2)]],
    device   float*          cn_stresses[[buffer(3)]],
    device   float*          ct_stresses[[buffer(4)]],
    device   float*          l_stresses [[buffer(5)]],
    constant float*          masses     [[buffer(6)]],
    device   int*            coord_nums [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint i = tid;
    if ((int)i >= p.num_particles) return;

    float3 grav = float3(p.grav_x, p.grav_y, p.grav_z);
    float3 f = float3(0.0f);
    if (p.gravity) f += grav * (9.81f * masses[i]);
    forces[i]  = st3(f);
    torques[i] = st3(float3(0.0f));

    for (int k=0; k<9; k++) {
        cn_stresses[i*9+k] = 0.0f;
        ct_stresses[i*9+k] = 0.0f;
        l_stresses[i*9+k]  = 0.0f;
    }
    coord_nums[i] = 0;
}

// ─────────────────────────────────────────────────────────
//  KERNEL: clear_interactions
// ─────────────────────────────────────────────────────────
kernel void clear_interactions(
    device int&              num_particles_ref [[buffer(0)]],
    device packed_float3*    old_interactions  [[buffer(1)]],
    device packed_float3*    new_interactions  [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int num_particles = num_particles_ref;
    int combis = num_particles * (num_particles + 1) / 2;
    uint i = tid;
    if ((int)i >= combis) return;

    old_interactions[i] = new_interactions[i];
    new_interactions[i] = packed_float3(0.0f, 0.0f, 0.0f);
}

// ─────────────────────────────────────────────────────────
//  SHARED PAIR PHYSICS  (used by all three pair kernels)
// ─────────────────────────────────────────────────────────

// process_pair: runs interaction_check → rod_rod_separation →
// rod_rod_collision_force → lubrication_forces for one (ii, jj) pair.
// new_int_buf: pass old_interactions for in-place update (2D/1D kernels)
//              or new_interactions for double-buffered update (cell kernel).
inline void process_pair(
    int ii, int jj, int N,
    constant PairParams&    p,
    device   packed_float3* CoMs,
    device   packed_float3* oris,
    constant float*         shaft_lengths,
    constant float*         radii,
    constant float*         masses,
    constant float*         avg_inerts,
    device   int*           coord_num,
    constant float*         fric_coefs,
    device   packed_float3* tvels,
    device   packed_float3* avels,
    device   packed_float3* forces,
    device   packed_float3* torques,
    device   float*         cn_stresses,
    device   float*         ct_stresses,
    device   float*         l_stresses,
    device   packed_float3* old_interactions,
    device   packed_float3* new_int_buf,
    device   packed_float3* mom_ins,
    device   float*         min_dt_cont,
    device   float*         min_dt_force,
    device   float*         min_dt_torque,
    constant PairConsts*    pair_consts,
    float3                  sys,
    int3                    pb)
{
    float3 rel_vel_LEBC_corr = float3(0.0f);
    float3 r_A_eff, r_B_eff;
    bool   interacting = true;

    interaction_check(
        ld3(CoMs[ii]), ld3(CoMs[jj]),
        shaft_lengths[ii], shaft_lengths[jj], radii[ii], radii[jj],
        sys, pb,
        p.max_sep, p.lub_toggle, p.LEBC_shift, p.LEBC_velo,
        rel_vel_LEBC_corr, r_A_eff, r_B_eff, interacting);

    if (!interacting) return;

    float  sep;
    float3 r_c, dir;
    rod_rod_separation(ld3(oris[ii]), ld3(oris[jj]),
                       shaft_lengths[ii], shaft_lengths[jj],
                       radii[ii], radii[jj],
                       sep, r_A_eff, r_B_eff, r_c, dir,
                       sys, pb);

    int        int_id = int_index(ii, jj, N);
    PairConsts pc     = pair_consts[int_id];

    rod_rod_collision_force(
        radii[ii], radii[jj], sep,
        r_A_eff, r_B_eff, r_c,
        pc.kn_eff, pc.M_eff, pc.en_eff, pc.et_eff, pc.t_c, pc.dc_n,
        avg_inerts[ii], avg_inerts[jj],
        &coord_num[ii], &coord_num[jj],
        fric_coefs[ii], fric_coefs[jj],
        ld3(tvels[ii]), ld3(tvels[jj]),
        ld3(avels[ii]), ld3(avels[jj]),
        dir,
        &forces[ii],  &forces[jj],
        &torques[ii], &torques[jj],
        cn_stresses, cn_stresses,
        ii, jj,
        ct_stresses, ct_stresses,
        ld3(old_interactions[int_id]),
        &new_int_buf[int_id],
        p.dt, p.contact_toggle, p.friction_toggle,
        rel_vel_LEBC_corr, min_dt_cont, p.gen_phase);

    if (p.lub_toggle) {
        lubrication_forces(
            radii[ii], radii[jj], sep,
            r_A_eff, r_B_eff, r_c,
            shaft_lengths[ii], shaft_lengths[jj],
            ld3(oris[ii]), ld3(oris[jj]),
            ld3(tvels[ii]), ld3(tvels[jj]),
            ld3(avels[ii]), ld3(avels[jj]),
            dir, rel_vel_LEBC_corr,
            p.viscosity, p.min_sep, p.max_sep,
            p.ee_manual_weight, p.ss_manual_weight, p.es_manual_weight,
            p.lub_toggle,
            &forces[ii], &forces[jj],
            &torques[ii], &torques[jj],
            l_stresses, l_stresses,
            ii, jj,
            ld3(mom_ins[ii]), ld3(mom_ins[jj]),
            masses[ii], masses[jj],
            min_dt_force, min_dt_torque);
    }
}

// ─────────────────────────────────────────────────────────
//  KERNEL: pair_interactions (2D dispatch)
// ─────────────────────────────────────────────────────────
kernel void pair_interactions(
    constant PairParams&    p              [[buffer(0)]],
    device   packed_float3* CoMs           [[buffer(1)]],
    device   packed_float3* oris           [[buffer(2)]],
    constant float*         shaft_lengths  [[buffer(3)]],
    constant float*         radii          [[buffer(4)]],
    constant float*         masses         [[buffer(5)]],
    constant float*         avg_inerts     [[buffer(6)]],
    device   int*           coord_num      [[buffer(7)]],
    // slots 8,9,10 (kns,ens,ets) unused — values come from pair_consts [[buffer(25)]]
    constant float*         fric_coefs     [[buffer(11)]],
    device   packed_float3* tvels          [[buffer(12)]],
    device   packed_float3* avels          [[buffer(13)]],
    device   packed_float3* forces         [[buffer(14)]],
    device   packed_float3* torques        [[buffer(15)]],
    device   float*         cn_stresses    [[buffer(16)]],
    device   float*         ct_stresses    [[buffer(17)]],
    device   float*         l_stresses     [[buffer(18)]],
    device   packed_float3* old_interactions [[buffer(19)]],
    device   packed_float3* new_interactions [[buffer(20)]],
    device   packed_float3* mom_ins        [[buffer(21)]],
    device   float*         min_dt_cont    [[buffer(22)]],
    device   float*         min_dt_force   [[buffer(23)]],
    device   float*         min_dt_torque  [[buffer(24)]],
    constant PairConsts*    pair_consts    [[buffer(25)]],
    uint2 gid [[thread_position_in_grid]])
{
    int ii = (int)gid.x;
    int jj = (int)gid.y;
    int N  = p.num_particles;

    if (ii >= N || jj >= N || ii >= jj) return;

    int3   pb  = int3(p.pb_x, p.pb_y, p.pb_z);
    float3 sys = float3(p.sys_x, p.sys_y, p.sys_z);

    process_pair(ii, jj, N, p, CoMs, oris, shaft_lengths, radii, masses, avg_inerts,
        coord_num, fric_coefs, tvels, avels, forces, torques,
        cn_stresses, ct_stresses, l_stresses,
        old_interactions, old_interactions,
        mom_ins, min_dt_cont, min_dt_force, min_dt_torque,
        pair_consts, sys, pb);
}

// ─────────────────────────────────────────────────────────
//  KERNEL: pair_interactions_1D
//  1D dispatch over N*(N-1)/2 unique pairs only.
//  Uses PairConsts precomputed buffer (item 1) to avoid
//  log/sqrt per active pair.
// ─────────────────────────────────────────────────────────
kernel void pair_interactions_1D(
    constant PairParams&     p              [[buffer(0)]],
    device   packed_float3*  CoMs           [[buffer(1)]],
    device   packed_float3*  oris           [[buffer(2)]],
    constant float*          shaft_lengths  [[buffer(3)]],
    constant float*          radii          [[buffer(4)]],
    constant float*          masses         [[buffer(5)]],
    constant float*          avg_inerts     [[buffer(6)]],
    device   int*            coord_num      [[buffer(7)]],
    // buffers 8,9,10 (kns,ens,ets) not needed — values come from pair_consts
    constant float*          fric_coefs     [[buffer(11)]],
    device   packed_float3*  tvels          [[buffer(12)]],
    device   packed_float3*  avels          [[buffer(13)]],
    device   packed_float3*  forces         [[buffer(14)]],
    device   packed_float3*  torques        [[buffer(15)]],
    device   float*          cn_stresses    [[buffer(16)]],
    device   float*          ct_stresses    [[buffer(17)]],
    device   float*          l_stresses     [[buffer(18)]],
    device   packed_float3*  old_interactions [[buffer(19)]],
    device   packed_float3*  new_interactions [[buffer(20)]],
    device   packed_float3*  mom_ins        [[buffer(21)]],
    device   float*          min_dt_cont    [[buffer(22)]],
    device   float*          min_dt_force   [[buffer(23)]],
    device   float*          min_dt_torque  [[buffer(24)]],
    constant PairConsts*     pair_consts    [[buffer(25)]],
    uint tid [[thread_position_in_grid]])
{
    int N = p.num_particles;
    // Map 1D tid → (ii, jj) using triangular inverse formula.
    // Row i contains (N-1-i) entries; cumulative before row i = i*(2N-i-1)/2.
    float fN   = (float)N;
    float ftid = (float)tid;
    int ii = (int)((2.0f*fN - 1.0f - sqrt((2.0f*fN - 1.0f)*(2.0f*fN - 1.0f) - 8.0f*ftid)) * 0.5f);
    // Clamp for floating-point rounding (at most 1 step off)
    if (ii > 0 && ii*(2*N-ii-1)/2 > (int)tid) ii--;
    if ((ii+1) < N-1 && (ii+1)*(2*N-ii-2)/2 <= (int)tid) ii++;
    int row_start = ii * (2*N - ii - 1) / 2;
    int jj = (int)tid - row_start + ii + 1;

    if (ii >= N || jj >= N || ii >= jj) return;

    int3   pb  = int3(p.pb_x, p.pb_y, p.pb_z);
    float3 sys = float3(p.sys_x, p.sys_y, p.sys_z);

    process_pair(ii, jj, N, p, CoMs, oris, shaft_lengths, radii, masses, avg_inerts,
        coord_num, fric_coefs, tvels, avels, forces, torques,
        cn_stresses, ct_stresses, l_stresses,
        old_interactions, old_interactions,
        mom_ins, min_dt_cont, min_dt_force, min_dt_torque,
        pair_consts, sys, pb);
}

// ─────────────────────────────────────────────────────────
//  KERNEL: body_interactions (1D dispatch)
// ─────────────────────────────────────────────────────────
kernel void body_interactions(
    constant BodyParams&    p             [[buffer(0)]],
    device   packed_float3* CoMs          [[buffer(1)]],
    device   packed_float3* oris          [[buffer(2)]],
    device   packed_float3* endpoints1    [[buffer(3)]],
    device   packed_float3* endpoints2    [[buffer(4)]],
    constant float*         radii         [[buffer(5)]],
    constant float*         shafts        [[buffer(6)]],
    constant float*         masses        [[buffer(7)]],
    device   packed_float3* tvels         [[buffer(8)]],
    device   packed_float3* avels         [[buffer(9)]],
    device   packed_float3* forces        [[buffer(10)]],
    device   packed_float3* torques       [[buffer(11)]],
    constant float*         velocity_prof [[buffer(12)]],
    constant float*         gradient_prof [[buffer(13)]],
    constant int*           coord_nums    [[buffer(14)]],
    uint tid [[thread_position_in_grid]])
{
    uint i = tid;
    if ((int)i >= p.num_particles) return;

    bool torque_drag_only = false;
    if (p.gen_phase && coord_nums[i] != 0) torque_drag_only = true;

    if (p.drag_toggle) {
        if (!torque_drag_only) {
            float3 df = drag_force_calc(
                ld3(CoMs[i]), ld3(tvels[i]), ld3(oris[i]),
                radii[i], shafts[i],
                velocity_prof, gradient_prof, p.num_bins, p.bin_size, p.viscosity);
            // atomic add to force (single thread per particle, no races here)
            float3 cf = ld3(forces[i]);
            forces[i] = st3(cf + df);
        }
        float3 ht = h_torque_calc(
            ld3(CoMs[i]), ld3(avels[i]),
            radii[i], shafts[i],
            ld3(endpoints1[i]), ld3(endpoints2[i]),
            gradient_prof, p.num_bins, p.bin_size, p.viscosity);
        float3 ct = ld3(torques[i]);
        torques[i] = st3(ct + ht);
    }

    if (p.lift_toggle) {
        float3 lf = lift_force_calc(
            ld3(CoMs[i]), ld3(tvels[i]), ld3(oris[i]),
            radii[i], shafts[i],
            velocity_prof, gradient_prof, p.num_bins, p.bin_size,
            p.viscosity, p.fluid_density);
        float3 cf = ld3(forces[i]);
        forces[i] = st3(cf + lf);
    }
}

// ─────────────────────────────────────────────────────────
//  KERNEL: body_interactions_kns  (version with kns/ens arrays for wall)
// ─────────────────────────────────────────────────────────
kernel void body_interactions_full(
    constant BodyParams&    p             [[buffer(0)]],
    device   packed_float3* CoMs          [[buffer(1)]],
    device   packed_float3* oris          [[buffer(2)]],
    device   packed_float3* endpoints1    [[buffer(3)]],
    device   packed_float3* endpoints2    [[buffer(4)]],
    constant float*         radii         [[buffer(5)]],
    constant float*         shafts        [[buffer(6)]],
    constant float*         masses        [[buffer(7)]],
    constant float*         kns           [[buffer(8)]],
    constant float*         ens           [[buffer(9)]],
    device   packed_float3* tvels         [[buffer(10)]],
    device   packed_float3* avels         [[buffer(11)]],
    device   packed_float3* forces        [[buffer(12)]],
    device   packed_float3* torques       [[buffer(13)]],
    constant float*         velocity_prof [[buffer(14)]],
    constant float*         gradient_prof [[buffer(15)]],
    constant int*           coord_nums    [[buffer(16)]],
    uint tid [[thread_position_in_grid]])
{
    uint i = tid;
    if ((int)i >= p.num_particles) return;

    bool torque_drag_only = false;
    if (p.gen_phase && coord_nums[i] != 0) torque_drag_only = true;

    if (p.drag_toggle) {
        if (!torque_drag_only) {
            float3 df = drag_force_calc(
                ld3(CoMs[i]), ld3(tvels[i]), ld3(oris[i]),
                radii[i], shafts[i],
                velocity_prof, gradient_prof, p.num_bins, p.bin_size, p.viscosity);
            float3 cf = ld3(forces[i]);
            forces[i] = st3(cf + df);
        }
        float3 ht = h_torque_calc(
            ld3(CoMs[i]), ld3(avels[i]),
            radii[i], shafts[i],
            ld3(endpoints1[i]), ld3(endpoints2[i]),
            gradient_prof, p.num_bins, p.bin_size, p.viscosity);
        float3 ct = ld3(torques[i]);
        torques[i] = st3(ct + ht);
    }

    if (p.lift_toggle) {
        float3 lf = lift_force_calc(
            ld3(CoMs[i]), ld3(tvels[i]), ld3(oris[i]),
            radii[i], shafts[i],
            velocity_prof, gradient_prof, p.num_bins, p.bin_size,
            p.viscosity, p.fluid_density);
        float3 cf = ld3(forces[i]);
        forces[i] = st3(cf + lf);
    }
}

// ─────────────────────────────────────────────────────────
//  KERNEL: integrate_positions
// ─────────────────────────────────────────────────────────
kernel void integrate_positions(
    constant IntegrateParams& p             [[buffer(0)]],
    device   packed_float3*   forces        [[buffer(1)]],
    device   packed_float3*   CoMs          [[buffer(2)]],
    device   packed_float3*   tvels         [[buffer(3)]],
    device   packed_float3*   taccs         [[buffer(4)]],
    constant float*           masses        [[buffer(5)]],
    device   packed_float3*   torques       [[buffer(6)]],
    device   packed_float3*   oris          [[buffer(7)]],
    device   packed_float3*   endpoints1    [[buffer(8)]],
    device   packed_float3*   endpoints2    [[buffer(9)]],
    constant float*           shaft_lengths [[buffer(10)]],
    device   packed_float3*   avels         [[buffer(11)]],
    device   packed_float3*   aaccs         [[buffer(12)]],
    device   packed_float3*   moi           [[buffer(13)]],
    uint tid [[thread_position_in_grid]])
{
    uint i = tid;
    if ((int)i >= p.num_particles) return;

    int3   pb  = int3(p.pb_x, p.pb_y, p.pb_z);
    float3 sys = float3(p.sys_x, p.sys_y, p.sys_z);

    float3 CoM    = ld3(CoMs[i]);
    float3 tvelo  = ld3(tvels[i]);
    float3 tacc_  = ld3(taccs[i]);
    float3 ori    = ld3(oris[i]);
    float3 ep1    = ld3(endpoints1[i]);
    float3 ep2    = ld3(endpoints2[i]);
    float3 avelo  = ld3(avels[i]);
    float3 aacc_  = ld3(aaccs[i]);
    float3 moiv   = ld3(moi[i]);

    update_position_msl(ld3(forces[i]), CoM, tvelo, tacc_, masses[i],
                         p.LEBC_shift, p.LEBC_velo, p.dt, sys, pb);
    update_orientation_msl(ld3(torques[i]), ori, ep1, ep2, CoM,
                            shaft_lengths[i], avelo, aacc_, moiv,
                            p.dt, p.allowRotation);

    CoMs[i]      = st3(CoM);
    tvels[i]     = st3(tvelo);
    taccs[i]     = st3(tacc_);
    oris[i]      = st3(ori);
    endpoints1[i]= st3(ep1);
    endpoints2[i]= st3(ep2);
    avels[i]     = st3(avelo);
    aaccs[i]     = st3(aacc_);
}

// ─────────────────────────────────────────────────────────
//  KERNEL: get_profiles
// ─────────────────────────────────────────────────────────
kernel void get_profiles(
    constant ProfileParams& p              [[buffer(0)]],
    device   packed_float3* CoMs           [[buffer(1)]],
    device   packed_float3* endpoints1     [[buffer(2)]],
    device   packed_float3* endpoints2     [[buffer(3)]],
    constant float*         volumes        [[buffer(4)]],
    device   float*         vol_frac_prof  [[buffer(5)]],
    device   float*         cn_stresses    [[buffer(6)]],
    device   float*         ct_stresses    [[buffer(7)]],
    device   float*         l_stresses     [[buffer(8)]],
    device   float*         cn_stress_prof [[buffer(9)]],
    device   float*         ct_stress_prof [[buffer(10)]],
    device   float*         l_stress_prof  [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    uint i = tid;
    if ((int)i >= p.num_particles) return;

    int3   pb  = int3(p.pb_x, p.pb_y, p.pb_z);
    float3 sys = float3(p.sys_x, p.sys_y, p.sys_z);

    add_volume_frac_contribution(
        ld3(CoMs[i]), ld3(endpoints1[i]), ld3(endpoints2[i]),
        volumes[i], p.num_bins, p.bin_size,
        vol_frac_prof, sys, pb);

    // Extract m[2] and m[6] for stress contribution
    float cn2 = cn_stresses[i*9+2], cn6 = cn_stresses[i*9+6];
    float ct2 = ct_stresses[i*9+2], ct6 = ct_stresses[i*9+6];
    float l2  = l_stresses[i*9+2],  l6  = l_stresses[i*9+6];

    float3 sys3 = sys;
    float3 CoM_ = ld3(CoMs[i]);
    int b    = (int)ceil(CoM_.z / p.bin_size) - 1;
    float bv = sys.x * sys.y * p.bin_size;
    if (b >= 0 && b < p.num_bins && bv > 0.0f) {
        atomic_add_f(&cn_stress_prof[b], ((cn2 + cn6) * 0.5f) / bv);
        atomic_add_f(&ct_stress_prof[b], ((ct2 + ct6) * 0.5f) / bv);
        atomic_add_f(&l_stress_prof[b],  ((l2  + l6)  * 0.5f) / bv);
    }
}

// ─────────────────────────────────────────────────────────
//  KERNEL: scale_CoMs
// ─────────────────────────────────────────────────────────
kernel void scale_CoMs(
    constant ScaleParams&   p    [[buffer(0)]],
    device   packed_float3* CoMs [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    uint i = tid;
    if ((int)i >= p.num_particles) return;
    float3 c = ld3(CoMs[i]);
    c.x *= p.scale; c.y *= p.scale; c.z *= p.scale;
    CoMs[i] = st3(c);
}

// ─────────────────────────────────────────────────────────
//  KERNEL: find_energy
// ─────────────────────────────────────────────────────────
kernel void find_energy(
    constant EnergyParams&  p    [[buffer(0)]],
    constant float*         masses [[buffer(1)]],
    device   packed_float3* tvels  [[buffer(2)]],
    device   packed_float3* moi    [[buffer(3)]],
    device   packed_float3* avels  [[buffer(4)]],
    device   float*         KE     [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint i = tid;
    if ((int)i >= p.num_particles) return;
    float3 tv = ld3(tvels[i]);
    float3 av = ld3(avels[i]);
    float3 mi = ld3(moi[i]);
    float ke_T = 0.5f * masses[i] * (tv.x*tv.x + tv.y*tv.y + tv.z*tv.z);
    float ke_A = 0.5f * (mi.x*av.x*av.x + mi.y*av.y*av.y + mi.z*av.z*av.z);
    KE[i] = ke_T + ke_A;
}

// ─────────────────────────────────────────────────────────
//  KERNEL: find_if_nonaffine_velocity_slow_enough
// ─────────────────────────────────────────────────────────
kernel void find_if_nonaffine_velocity_slow_enough(
    constant NonAffineParams& p           [[buffer(0)]],
    device   packed_float3*   trans_velocs[[buffer(1)]],
    device   packed_float3*   CoMs        [[buffer(2)]],
    constant float*           vel_profile [[buffer(3)]],
    constant float*           grad_profile[[buffer(4)]],
    device   float*           global_max  [[buffer(5)]],
    constant float*           radii       [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint i = tid;
    if ((int)i >= p.num_particles) return;

    float3 CoM = ld3(CoMs[i]);
    int b = (int)floor(CoM.z / p.bin_size);
    b = clamp(b, 0, p.num_grad_bins-1);
    float g = grad_profile[b];
    float velo = vel_profile[b+1] - (g * (p.bin_size + p.bin_size*b - CoM.z));
    float3 fv = float3(velo, 0.0f, 0.0f);

    float3 na_v = ld3(trans_velocs[i]) - fv;
    float  mag_na = mag3(na_v);

    float LHS = 1e-3f * radii[i] * p.characteristic_shearrate;
    float diff = mag_na - LHS;
    atomic_max_f(global_max, diff);
}

// ─────────────────────────────────────────────────────────
//  CELL LIST STRUCTS AND KERNELS
// ─────────────────────────────────────────────────────────

struct CellListParams {
    int   num_particles;
    int   nx, ny, nz;
    float cell_size;
    float sys_x, sys_y, sys_z;
    int   pb_x, pb_y, pb_z;
    int   max_cells;
};

// ── kernel 1: assign each particle its cell id and count per cell ──
kernel void cell_assign_count(
    constant CellListParams& cl         [[buffer(0)]],
    device   packed_float3*  CoMs       [[buffer(1)]],
    device   int*            cell_id    [[buffer(2)]],
    device   int*            cell_count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    int i = (int)tid;
    if (i >= cl.num_particles) return;

    float3 pos = ld3(CoMs[i]);
    int cx = (int)(pos.x / cl.cell_size);
    int cy = (int)(pos.y / cl.cell_size);
    int cz = (int)(pos.z / cl.cell_size);
    cx = clamp(cx, 0, cl.nx - 1);
    cy = clamp(cy, 0, cl.ny - 1);
    cz = clamp(cz, 0, cl.nz - 1);
    int cid = cx + cy * cl.nx + cz * cl.nx * cl.ny;
    cell_id[i] = cid;
    atomic_add_i(&cell_count[cid], 1);
}

// ── kernel 2: scatter particles into sorted list ──
// Call after CPU builds cell_start from cell_count.
// cell_offset must be zeroed before dispatch (reuse cell_count buffer).
kernel void cell_fill_sorted(
    constant CellListParams& cl           [[buffer(0)]],
    device   int*            cell_id      [[buffer(1)]],
    device   int*            cell_start   [[buffer(2)]],
    device   int*            cell_offset  [[buffer(3)]],
    device   int*            sorted_parts [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    int i = (int)tid;
    if (i >= cl.num_particles) return;

    int cid  = cell_id[i];
    int slot = atomic_fetch_add_explicit(
        (device atomic_int*)&cell_offset[cid], 1, memory_order_relaxed);
    sorted_parts[cell_start[cid] + slot] = i;
}

// ── kernel 3: pairwise interactions using cell list (1 thread per particle ii) ──
kernel void pair_interactions_cell(
    constant CellListParams&  cl              [[buffer(0)]],
    constant PairParams&      p               [[buffer(1)]],
    device   packed_float3*   CoMs            [[buffer(2)]],
    device   packed_float3*   oris            [[buffer(3)]],
    constant float*           shaft_lengths   [[buffer(4)]],
    constant float*           radii           [[buffer(5)]],
    constant float*           masses          [[buffer(6)]],
    constant float*           avg_inerts      [[buffer(7)]],
    device   int*             coord_num       [[buffer(8)]],
    constant float*           fric_coefs      [[buffer(9)]],
    device   packed_float3*   tvels           [[buffer(10)]],
    device   packed_float3*   avels           [[buffer(11)]],
    device   packed_float3*   forces          [[buffer(12)]],
    device   packed_float3*   torques         [[buffer(13)]],
    device   float*           cn_stresses     [[buffer(14)]],
    device   float*           ct_stresses     [[buffer(15)]],
    device   float*           l_stresses      [[buffer(16)]],
    device   packed_float3*   old_interactions[[buffer(17)]],
    device   packed_float3*   new_interactions[[buffer(18)]],
    device   packed_float3*   mom_ins         [[buffer(19)]],
    device   float*           min_dt_cont     [[buffer(20)]],
    device   float*           min_dt_force    [[buffer(21)]],
    device   float*           min_dt_torque   [[buffer(22)]],
    constant PairConsts*      pair_consts     [[buffer(23)]],
    constant int*             cell_id         [[buffer(24)]],
    constant int*             cell_count      [[buffer(25)]],
    constant int*             cell_start      [[buffer(26)]],
    constant int*             sorted_parts    [[buffer(27)]],
    uint tid [[thread_position_in_grid]])
{
    int ii = (int)tid;
    int N  = p.num_particles;
    if (ii >= N) return;

    int3   pb  = int3(p.pb_x, p.pb_y, p.pb_z);
    float3 sys = float3(p.sys_x, p.sys_y, p.sys_z);

    // Find which cell particle ii lives in
    float3 pos_ii = ld3(CoMs[ii]);
    int cx = clamp((int)(pos_ii.x / cl.cell_size), 0, cl.nx - 1);
    int cy = clamp((int)(pos_ii.y / cl.cell_size), 0, cl.ny - 1);
    int cz = clamp((int)(pos_ii.z / cl.cell_size), 0, cl.nz - 1);

    // Walk 3×3×3 neighbourhood
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int nx_ = cx + dx;
        int ny_ = cy + dy;
        int nz_ = cz + dz;

        if (pb.x) { nx_ = ((nx_ % cl.nx) + cl.nx) % cl.nx; }
        else       { if (nx_ < 0 || nx_ >= cl.nx) continue;  }
        if (pb.y) { ny_ = ((ny_ % cl.ny) + cl.ny) % cl.ny; }
        else       { if (ny_ < 0 || ny_ >= cl.ny) continue;  }
        if (pb.z) { nz_ = ((nz_ % cl.nz) + cl.nz) % cl.nz; }
        else       { if (nz_ < 0 || nz_ >= cl.nz) continue;  }

        int ncell = nx_ + ny_ * cl.nx + nz_ * cl.nx * cl.ny;
        int count = cell_count[ncell];
        int start = cell_start[ncell];

        for (int k = 0; k < count; ++k) {
            int jj = sorted_parts[start + k];
            if (jj <= ii) continue;   // each pair processed once (ii < jj)

            process_pair(ii, jj, N, p, CoMs, oris, shaft_lengths, radii, masses, avg_inerts,
                coord_num, fric_coefs, tvels, avels, forces, torques,
                cn_stresses, ct_stresses, l_stresses,
                old_interactions, new_interactions,
                mom_ins, min_dt_cont, min_dt_force, min_dt_torque,
                pair_consts, sys, pb);
        }
    }
}
