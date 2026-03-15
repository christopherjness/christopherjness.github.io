# metalRoDS — Claude Context

## What this is

A Discrete Element Method (DEM) simulation of rod-shaped particles (spherocylinders) on Apple GPU via Metal. It is a direct translation of an original CUDA implementation; the physics are intentionally identical.

## Source files

| File | Role |
|------|------|
| `main_metal.mm` | Objective-C++ host: setup, config parsing, GPU buffer management, simulation loop |
| `rods_kernels.metal` | MSL GPU kernels: all per-particle and per-pair force/integration kernels |
| `config_profile.txt` | Runtime config (plain text key=value); read at startup |
| `physics.tex` | **Authoritative physics reference** — any kernel changes must match the math here |
| `particles_profile.txt` | Optional polydisperse particle size input |

Output files (`output_profile_*`) are simulation outputs, not sources.

## Build

```
make          # builds rods_kernels.metallib then metal_sim
make clean    # removes .air, .metallib, metal_sim
```

Two-step Metal compile: `.metal` → `.air` (xcrun metal) → `.metallib` (xcrun metallib). The `.metallib` path is read at runtime from the config file (`metallib_path = rods_kernels.metallib`).

**Dependencies** (Homebrew, Apple Silicon paths):
- Eigen3: `/opt/homebrew/include/eigen3`
- nlohmann/json: `/opt/homebrew/include`
- Frameworks: Metal, Foundation, CoreGraphics

## Key gotcha — header order in main_metal.mm

`struct float3` and `struct int3` **must** be defined before any Metal/simd headers are imported, or there will be type conflicts. This is intentional and must be preserved.

## Physics

Particles are **spherocylinders** (cylindrical shaft + hemispherical caps), described by position, orientation unit vector, and angular velocity. Forces:

- **Contact**: linear spring-dashpot (normal + tangential), with optional friction
- **Lubrication**: sphere-sphere, end-end, end-side resistance, with manual weights (`ss/es/ee_manual_weight`)
- **Drag / Lift**: fluid coupling
- **Gravity**: optional

Integration uses explicit Euler with optional dynamic timestep. Supports **Lees-Edwards boundary conditions** (LEBC) for shear flow and periodic boundaries in all three axes.

All force models are documented in `physics.tex`. Kernel changes should be verified against that document.

## Config parameters (key ones)

- `num_particles`, `aspect` — particle count and rod aspect ratio
- `radius_1/2`, `prop_1/2` — bidisperse size distribution
- `kn`, `en`, `et`, `friction_coef` — contact stiffness, restitution, friction
- `fluid_viscosity`, `fluid_density` — fluid properties
- `lub_min_sep`, `lub_max_sep` — lubrication cutoffs
- `dt`, `dynamic_dt` — timestep control
- `TPB`, `TPB2`, `TPB3` — Metal threadgroup sizes (threads per block)
- `contact_toggle`, `friction_toggle`, `lub_toggle`, `drag_toggle`, `lift_toggle` — enable/disable physics terms

## Platform constraints

- macOS only (Metal framework)
- Apple Silicon assumed (`-march=native`, Homebrew at `/opt/homebrew`)
- Performance-sensitive: `-O3 -ffast-math -flto` on host; `-O2 -ffast-math` on Metal shaders. Do not weaken these.
