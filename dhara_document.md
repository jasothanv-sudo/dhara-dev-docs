# DHARA — Documentation

A modular Python framework for time-dependent PDE solvers in fluid dynamics and related physics, covering four modules: **Compressible**, **Convection**, **Rayleigh–Bénard**, and **Quantum**. The same core — grids, time integrators, boundary handling, and data I/O — is reused across all of them, and every run can execute on CPU or GPU, serially or in parallel, selected entirely from a plain-text parameter file.

---

## Table of Contents

- **Chapter 0** — [Installation Guide](#ch0)
- **Chapter 1** — [Overview and Architecture](#ch1)
- **Chapter 2** — [Theory: Compressible Module](#ch2)
- **Chapter 3** — [Theory: Convection Module](#ch3)
- **Chapter 4** — [Theory: Rayleigh–Bénard Module](#ch4)
- **Chapter 5** — [Theory: Quantum Module](#ch5)
- **Chapter 6** — [Numerical Framework](#ch6)
- **Chapter 7** — [Available Problems and Sub-Problems](#ch7)
- **Chapter 8** — [Configuring `para.py`](#ch8)
- **Chapter 9** — [Worked Example: Taylor–Green Vortex on GPU](#ch9)

---

<a id="ch0"></a>

## Chapter 0 — Installation Guide

DHARA is built on the scientific-Python stack. A basic CPU install needs only Conda and a few libraries. GPU execution (CuPy) and parallel execution (MPI) are optional and can be added independently, in any order, on top of the base install.

### 0.1 System Requirements

| Requirement | Notes |
|---|---|
| **Operating system** | Linux or macOS recommended (typical for GPU/cluster runs). Windows works for serial CPU/GPU runs; parallel runs on Windows need Microsoft MPI (see §0.6). |
| **Python** | 3.8 or newer (3.10 recommended). |
| **Core libraries** | `numpy`, `scipy`, `h5py`, `imageio`. |
| **GPU (optional)** | An NVIDIA GPU with a compatible CUDA toolkit and a matching `cupy` build. |
| **Parallel (optional)** | An MPI implementation plus `mpi4py`; parallel field output additionally needs an MPI-enabled `h5py` build. |
| **Disk** | Simulations write HDF5 field snapshots and global-quantity files to the run directory's `output/` folder; size depends on grid and save frequency. |

Simulations are launched from inside a specific example directory (e.g. `compressible/2d/`), which the run scripts expect — see §0.7.

### 0.2 Install Conda and Create an Environment

If you do not already have Conda, install **Miniconda** for your platform from the official page:

- <https://docs.conda.io/en/latest/miniconda.html>

Then create and activate a dedicated environment:

```bash
conda create -n dhara_env python=3.10 -y
```

```bash
conda activate dhara_env
```

Keeping DHARA in its own environment avoids version conflicts and makes it easy to add the optional CUDA and MPI components in isolation.

### 0.3 Install the Core Python Libraries

From the project root (the folder containing `requirements.txt`), install the required packages:

```bash
pip install -r requirements.txt
```

This installs `numpy`, `scipy`, `h5py`, and `imageio`. At this point DHARA can run **serial CPU** simulations. The GPU and parallel steps below are only needed for those capabilities.

### 0.4 Install the CUDA Toolkit (GPU only)

*Skip this section for CPU-only use.*

GPU execution uses [CuPy](https://cupy.dev/), which requires an NVIDIA CUDA toolkit installed on the system. Install a toolkit that matches your GPU driver:

- NVIDIA CUDA Toolkit downloads: <https://developer.nvidia.com/cuda-downloads>

Verify the toolkit is visible before installing CuPy:

```bash
nvcc --version
```

Note the reported CUDA version (e.g. 11.x or 12.x) — you will match CuPy to it in the next step.

### 0.5 Install CuPy (GPU only)

*Skip this section for CPU-only use.*

Install the CuPy wheel that matches your CUDA major version:

```bash
pip install cupy-cuda12x
```

Use `cupy-cuda11x` instead if your toolkit is CUDA 11. Verify the install and that a GPU is detected:

```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s) detected')"
```

To run on the GPU, set `device = 'GPU'` in `para.py` (and `device_rank` to select which GPU). No source changes are required — DHARA switches its array backend from NumPy to CuPy automatically.

### 0.6 Install MPI and mpi4py (Parallel Runs — optional)

*Skip this section for single-process runs.*

Parallel execution decomposes the domain across MPI processes. This requires two pieces: a system-level **MPI implementation**, and the Python binding **mpi4py**.

#### 0.6.1 Manually Download and Install an MPI Implementation

Install an MPI library by downloading it from the official source and following its installation instructions for your platform. Two widely used open-source implementations:

- **Open MPI** — <https://www.open-mpi.org/software/ompi/>
- **MPICH** — <https://www.mpich.org/downloads/>

> **Windows users:** Open MPI and MPICH are Linux/macOS-oriented. On Windows, install **Microsoft MPI (MS-MPI)** instead, from <https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi> — `mpi4py` builds against it.

After installation, confirm the MPI launcher is on your `PATH`:

```bash
mpirun --version
```

#### 0.6.2 Install mpi4py

With an MPI implementation present, install the Python binding:

```bash
pip install mpi4py
```

`mpi4py` compiles against the MPI installation found on your `PATH`, so §0.6.1 must be completed first.

> **Parallel field output:** DHARA writes field snapshots collectively using HDF5's MPI-IO driver during parallel runs. This requires an `h5py` built against a **parallel (MPI-enabled) HDF5** library. The default `pip install h5py` from §0.3 is serial-only. If you intend to save fields from parallel runs, install a parallel-enabled `h5py` build (for example via `conda install -c conda-forge "h5py=*=mpi*"`, or by building `h5py` from source with `HDF5_MPI=ON`).

#### 0.6.3 Launching Parallel Runs

Enable parallelism by setting `is_parallel = True` in `para.py`, then launch from the example directory, specifying the number of processes:

```bash
mpirun -np 4 python main.py
```

The domain is decomposed into slabs along the first axis, so the number of grid points along that axis should be divisible by the number of processes. Serial and parallel runs use the same `main.py` and `para.py` — only `is_parallel` and the launch command differ.

### 0.7 Running a Simulation

Simulations are launched from within a specific example directory, not the project root. For example, to run the 2D compressible case:

```bash
cd compressible/2d
```

```bash
python main.py
```

Each example folder contains a `para.py` (all configurable parameters — see Chapter 8) and a `main.py` (the driver). Output is written to the `output_dir` set in `para.py` (default `output/test/`), and a copy of `para.py` is saved alongside the results as `parameters_<n>.py` so every run is reproducible.

---

<a id="ch1"></a>

## Chapter 1 — Overview and Architecture

### 1.1 What DHARA Is

DHARA is a modular framework in which a single shared core — grids, time integrators, boundary handling, and data I/O — is reused across four physics modules. The framework is written against an array-backend abstraction that lets identical solver code run on **CPU (NumPy)** or **GPU (CuPy)**, and a domain-decomposition layer that lets the same code run **serially or in parallel (MPI)** — all selected from `para.py`, with no source changes.

The four modules:

| Module | `problem` value | Physics | Method |
|---|---|---|---|
| **Compressible** | `'euler'`, `'hydro'` | Compressible Euler / Navier–Stokes gas dynamics | Finite-volume, WENO/TENO reconstruction + Kurganov–Tadmor flux |
| **Convection** | `'convection'` | Compressible convection in a gravitationally stratified layer | Same FV core as Compressible, plus gravity and a polytropic base state |
| **Rayleigh–Bénard** | `'rbc_tdma'` | Incompressible Boussinesq thermal convection | Pressure-projection, spectral-in-x + TDMA-in-y Poisson solve |
| **Quantum** | `'quantum'` | Gross–Pitaevskii equation (BEC / quantum turbulence) | Spectral time-splitting (TSSP), complex fields |

Each module supports multiple spatial dimensions (1D/2D/3D, and 4D for Rayleigh–Bénard), selected by a single parameter.

### 1.2 Package Layout and the Anatomy of a Run

DHARA separates **what you edit** from **what you run against**. You never modify the core package; you configure a run directory that imports it.

**Run directories** (`compressible/`, `convection/`, `quantum/`, `rbc/`) each contain per-dimension example folders, and every example folder holds exactly two files:

- **`para.py`** — every configurable parameter for the run. This is the only file you normally edit (Chapter 8).
- **`main.py`** — the driver: reads `para.py`, builds the solver, sets initial conditions, and runs the time loop.

**The core package** (`DHARA/`) provides everything the driver imports, in three layers:

```
DHARA/
├── __init__.py        ← public API — exports every solver, grid, and utility class
├── init_cond/         ← initial-condition builders (init1D / init2D / init3D)
└── src/
    ├── univ/          ← universal machinery: grids, time evolution, data I/O, problem base
    ├── lib/           ← reusable numerics: reconstruction, fluxes, derivatives, Poisson solvers
    └── problem/       ← the physics modules: euler, hydro, convection, rbc_tdma, quantum
```

**Solver classes are built by composition.** A physics solver inherits a grid, the universal problem base, and the numerical mixins it needs. For example, the 2D compressible stack builds up as:

```
Grid2D + ProblemSetup + KTConvFlux   →  Euler2D
Euler2D + VisFlux2D                  →  Hydro2D
Hydro2D                              →  Convection2D
```

This is why the Convection module *is* the Compressible module plus a gravity term and a stratified base state (Chapter 3), and why fixing a bug in the flux routine improves every compressible-family solver at once.

**Execution flow** inside `main.py` is the same for every module:

1. Read every variable from `para.py` into a `params` dictionary.
2. Instantiate the solver for the chosen `problem` and `dimension` (e.g. `Hydro2D(params)`), allocating the grid and fields.
3. Apply initial conditions — from a built-in `sub_problem`, from an `input/*.h5` file, or by continuing a previous run (`profile`).
4. Create a `DataIO` object (output paths, save cadence, plotting).
5. Create a `Time_Evolution` object and call the appropriate time-advance driver.
6. Loop from `tinit` to `tfinal`, at each step saving any due output, advancing the solution, and printing a status line.

The time integrator is chosen per module: the compressible, convection, and Rayleigh–Bénard modules use SSP Runge–Kutta (`RK2`/`RK3`, via `scheme`); the quantum module uses its self-contained time-splitting step.

### 1.3 Execution Model: Device and Parallelism

Two orthogonal switches in `para.py` control *where* and *how* a simulation runs, without touching solver code.

**Device (CPU vs. GPU).** Every solver accesses arrays through a single backend handle: NumPy when `device = 'CPU'`, CuPy when `device = 'GPU'`. All array creation, math, and FFTs go through this handle, so identical code executes on either processor. `device_rank` selects which GPU on a multi-GPU node. Switching between CPU and GPU is purely a parameter change.

**Parallelism (MPI).** When `is_parallel = True`, the domain is split into slabs across MPI processes along the first axis — *x* for 1D–3D, *w* for 4D Rayleigh–Bénard. Each process owns its slab plus a layer of **ghost cells** mirroring its neighbours' edge data. The core handles the three concerns this introduces:

- **Halo exchange** — processes exchange edge cells into ghost regions each step (non-blocking point-to-point).
- **Global reductions** — integral diagnostics are summed across processes so printed status reflects the whole domain.
- **Collective output** — field snapshots are written to a single HDF5 file via the MPI-IO driver.

Because the two switches are independent, all four combinations — serial CPU, serial GPU, parallel CPU, parallel GPU — are available from the same source.

### 1.4 Output and Restart Model

Every run writes into the directory named by `output_dir` (default `output/test/`), created on startup with three products:

- **`fields/`** — HDF5 snapshots of the full solution (`<dim>D_<time>.h5`), written every `save_fields_dt`. These are both the primary scientific output and the checkpoints used for restarts.
- **Global-quantity files** — time series of integral diagnostics (energy, mass, module-specific quantities), sampled at `save_global_quantities_dt`.
- **`plots/`** — optional PNG frames and an assembled movie, when plotting is enabled and available for that module and dimension (Chapter 8, §8.6).

On startup the driver also copies the active `para.py` into the output directory as `parameters_<n>.py`, so each result set records the parameters that produced it. To resume a run, set `profile = 'continue'` and `tinit` to the time of an existing snapshot.

---

<a id="ch2"></a>

## Chapter 2 — Theory: Compressible Module

**`problem = 'hydro'` (viscous) / `'euler'` (inviscid)**

### 2.1 Governing Equations

The module solves the compressible Navier–Stokes equations in conservative form. With state vector $q = (\rho,\ \rho u,\ \rho v,\ E)$:

$$
\begin{aligned}
\frac{\partial \rho}{\partial t} + \nabla\cdot(\rho\mathbf{u}) &= 0 \\
\frac{\partial (\rho\mathbf{u})}{\partial t} + \nabla\cdot(\rho\,\mathbf{u}\otimes\mathbf{u} + p\mathbf{I}) &= \nabla\cdot\boldsymbol{\tau} + \mathbf{f} \\
\frac{\partial E}{\partial t} + \nabla\cdot\big[(E + p)\mathbf{u}\big] &= \nabla\cdot(\boldsymbol{\tau}\cdot\mathbf{u}) + \nabla\cdot(K\,\nabla T)
\end{aligned}
$$

Setting the viscous stress $\boldsymbol{\tau} = 0$ and conductivity $K = 0$ recovers the **Euler equations** (`problem = 'euler'`), on which the `hydro` solver is built.

### 2.2 Closure Relations

- **Equation of state (ideal gas):** $\ p = (\gamma - 1)\left(E - \tfrac{1}{2}\rho\lvert\mathbf{u}\rvert^2\right)\ $ and $\ p = R_{\text{gas}}\,\rho\,T$
- **Newtonian viscous stress:** $\ \boldsymbol{\tau} = \mu\left[\nabla\mathbf{u} + (\nabla\mathbf{u})^{\mathsf{T}} - \tfrac{2}{3}(\nabla\cdot\mathbf{u})\mathbf{I}\right]$
- **Transport coefficients:** $\ R_{\text{gas}} = \dfrac{1}{\gamma\,\mathrm{Ma}^2},\quad \mu = \dfrac{1}{\mathrm{Re}},\quad K = \dfrac{1}{\mathrm{Ma}^2(\gamma-1)\,\mathrm{Re}\,\mathrm{Pr}}$
- $\mathbf{f}$ is an optional stochastic forcing term (`forcing = True`), used to sustain turbulence.

### 2.3 Governing Parameters

| Parameter | Symbol | Meaning |
|---|---|---|
| `gamma` | γ | Ratio of specific heats |
| `Ma` | Ma | Mach number (sets the gas constant / compressibility) |
| `Re` | Re | Reynolds number (`hydro` only) |
| `Pr` | Pr | Prandtl number |

Equations are non-dimensionalized by reference density, length, and velocity; the flow speed relative to the sound speed is fixed by the Mach number. Typical cases: Taylor–Green vortex, Sod shock tube, forced compressible turbulence (Chapter 7).

---

<a id="ch3"></a>

## Chapter 3 — Theory: Convection Module

**`problem = 'convection'`**

### 3.1 Governing Equations

The convection module solves the **same compressible Navier–Stokes equations as Chapter 2**, with one physical addition: a gravitational body force acting on a stratified atmosphere. The vertical momentum and energy equations gain gravity source terms (gravity acts in the $-y$ direction):

$$
\begin{aligned}
\frac{\partial (\rho v)}{\partial t} + \cdots &= \cdots - \rho g \\
\frac{\partial E}{\partial t} + \cdots &= \cdots - \rho g\, v
\end{aligned}
$$

The solver inherits directly from the compressible module; the numerics (reconstruction, flux, viscous terms) are identical. The physics differs through gravity, a stratified base state, and wall boundaries.

### 3.2 Base State and Closure

The layer is initialized in **hydrostatic polytropic equilibrium** of index $m$, with an **adiabatic reference** state used for diagnostics:

$$
\begin{aligned}
\text{Hydrostatic:}\quad & T_0 = 1-\beta y, \quad \rho_0 = (1-\beta y)^{m}, \quad p_0 = R_{\text{gas}}(1-\beta y)^{m+1} \\
\text{Adiabatic:}\quad & T_{\mathrm{ad}} = 1-D y, \quad \rho_{\mathrm{ad}} = (1-D y)^{\alpha}, \quad p_{\mathrm{ad}} = R_{\text{gas}}(1-D y)^{\alpha+1}
\end{aligned}
$$

with adiabatic index $\alpha = 1/(\gamma-1)$, $\ \beta = \epsilon + D$, and polytropic index $m = D(1+\alpha)/\beta - 1$. The saved temperature is the **super-adiabatic** departure $T_{\mathrm{sa}} = T - T_{\mathrm{ad}}$, which measures the convective driving.

Transport and gravity are set by the control parameters (not by the Mach and Reynolds numbers):

$$
R_{\text{gas}} = \frac{1}{\epsilon D(\alpha+1)}, \qquad \mu = \sqrt{\frac{\mathrm{Pr}}{\mathrm{Ra}}}, \qquad g = \frac{1}{\epsilon}, \qquad K = \frac{1}{\epsilon D\sqrt{\mathrm{Ra}\,\mathrm{Pr}}}.
$$

### 3.3 Boundary Conditions and Parameters

Horizontal (*x*): periodic. Vertical (*y*): **no-slip walls** with **fixed temperatures** $T_b = 1$ (bottom), $T_t = 1 - \beta$ (top). Optional shear forcing and internal heating are available.

| Parameter | Symbol | Meaning |
|---|---|---|
| `gamma` | γ | Ratio of specific heats |
| `epsilon` | ε | Superadiabaticity (thermal driving) |
| `D` | D | Dissipation number (stratification depth) |
| `Ra` | Ra | Rayleigh number |
| `Pr` | Pr | Prandtl number |

> **Configuration note:** the gravitational body force is applied through the solver's forcing hook, so `forcing = True` is required for a physical convection run (Chapter 8, §8.5.2).

---

<a id="ch4"></a>

## Chapter 4 — Theory: Rayleigh–Bénard Module

**`problem = 'rbc_tdma'`**

### 4.1 Governing Equations

Unlike Chapters 2–3, this module solves the **incompressible Boussinesq equations**: buoyancy-driven flow with density variation retained only in the buoyancy term. With velocity $\mathbf{u}$ and temperature $T$:

$$
\begin{aligned}
\nabla\cdot\mathbf{u} &= 0 \\
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u} &= -\nabla p + \sqrt{\frac{\mathrm{Pr}}{\mathrm{Ra}}}\;\nabla^2\mathbf{u} + T\,\hat{\mathbf{y}} \\
\frac{\partial T}{\partial t} + (\mathbf{u}\cdot\nabla)T &= \frac{1}{\sqrt{\mathrm{Ra}\,\mathrm{Pr}}}\;\nabla^2 T
\end{aligned}
$$

$\hat{\mathbf{y}}$ is the vertical (anti-gravity) unit vector; the $T\,\hat{\mathbf{y}}$ term is the buoyancy coupling that drives convection.

### 4.2 Numerical Closure

Incompressibility is enforced by a **pressure-projection** step: the pressure Poisson equation is solved by Fourier transform in the horizontal (periodic) direction combined with a tridiagonal (TDMA) solve in the vertical — hence `rbc_tdma`. The temperature field evolved internally is the perturbation about the conductive profile; the saved temperature restores the linear background, $T = \theta + (1 - y)$.

Viscosity and thermal diffusivity follow directly from the two control parameters:

$$
\nu = \sqrt{\frac{\mathrm{Pr}}{\mathrm{Ra}}}, \qquad \kappa = \frac{1}{\sqrt{\mathrm{Ra}\,\mathrm{Pr}}}.
$$

### 4.3 Boundary Conditions and Parameters

Horizontal: periodic. Vertical: fixed temperature (hot below, cold above) with either **no-slip** or **free-slip** walls (`sub_problem`). The vertical grid is typically clustered toward the walls (`spacing_type = 'non-uniform'`, Chapter 8).

| Parameter | Symbol | Meaning |
|---|---|---|
| `Ra` | Ra | Rayleigh number (buoyancy vs. dissipation) |
| `Pr` | Pr | Prandtl number (momentum vs. thermal diffusivity) |

Convection sets in above a critical Rayleigh number (order $10^3$, boundary-dependent); the Rayleigh number is chosen relative to this threshold to select conductive or convective regimes. Available in 2D, 3D, and 4D.

---

<a id="ch5"></a>

## Chapter 5 — Theory: Quantum Module

**`problem = 'quantum'`**

### 5.1 Governing Equation

The quantum module solves the **Gross–Pitaevskii equation (GPE)** for a complex wavefunction $\psi$, the mean-field model of a Bose–Einstein condensate / superfluid:

$$
i\,\frac{\partial \psi}{\partial t} = -\alpha\,\nabla^2\psi + V\psi + g_0\,\lvert\psi\rvert^2\psi
$$

- $-\alpha\,\nabla^2\psi$ — dispersion (quantum kinetic term),
- $V\psi$ — external potential,
- $g_0\,\lvert\psi\rvert^2\psi$ — nonlinear self-interaction.

Setting $g_0 = 0$ reduces the solver to the **linear Schrödinger equation**; nonzero $g_0$ (attractive or repulsive) gives the full condensate dynamics.

### 5.2 Method and Normalization

The equation is advanced by a **time-splitting spectral (TSSP)** scheme: the nonlinear/potential part is applied as an exact pointwise phase rotation over half-steps, and the dispersion part is advanced exactly in Fourier space. The domain is periodic and the grid uniform. The density $\lvert\psi\rvert^2$ integrates to a conserved norm, tracked each step as a diagnostic.

### 5.3 Parameters

| Parameter | Symbol | Meaning |
|---|---|---|
| `alpha` | α | Coefficient of the dispersion (Laplacian) term |
| `g0` | g₀ | Nonlinear interaction strength (0 ⇒ linear) |
| `potential_type` | — | External potential: `none` or `harmonic` |

For quantum-turbulence studies, the module adds optional spectral-shell **forcing** and **hyper-/hypo-viscous dissipation** (large- and small-scale damping), configured in `para.py` (Chapter 8, §8.5.4). Available in 1D, 2D, and 3D.

---

<a id="ch6"></a>

## Chapter 6 — Numerical Framework

DHARA's physics modules are assembled from a shared library of grids, spatial-discretization schemes, elliptic solvers, and time integrators. This chapter documents the grid system in full and catalogs the available methods; the mathematical details of each scheme are outside the scope of this manual.

### 6.1 Grids

#### 6.1.1 Construction

Grids are provided in one, two, three, and four dimensions (`Grid1D`–`Grid4D`). A grid is defined entirely by parameters in `para.py`:

- `dimension` — selects the grid class,
- `N` — number of real cells per axis,
- `L_min`, `L_max` — domain bounds per axis.

The grid builds the coordinate arrays (*x*, *y*, *z*, *w*), the cell spacings, and all working ("scratch") arrays that the solver fields occupy. Every array is allocated on the active backend (NumPy or CuPy), so the grid is also where the CPU/GPU choice takes physical effect.

#### 6.1.2 Layout

All finite-volume modules (compressible, convection) and the Rayleigh–Bénard module use a **collocated** layout — every variable lives at the same cell centers. The quantum module uses a **nodal** spectral layout suited to its FFT-based solver. The layout is fixed internally per module and is not a user choice (the `layout_type` entry in `para.py` is not consulted — Chapter 8, §8.1).

#### 6.1.3 Uniform and Non-Uniform Spacing

Two grid spacings are available, set by `spacing_type`:

- **`'uniform'`** — constant cell size on every axis.
- **`'non-uniform'`** — **tangent-hyperbolic clustering in the *y*-direction only**; the *x*-axis (and *z*, *w*) remain uniform. The clustering concentrates cells toward the top and bottom of the domain, resolving wall boundary layers in convection and Rayleigh–Bénard runs.

The clustering strength is controlled by `beta` (larger `beta` ⇒ stronger clustering at the walls). For a non-uniform grid the solver also stores the coordinate **Jacobian**, so that derivatives and volume integrals are computed correctly on the stretched mesh. Because clustering is vertical only, the horizontal directions stay uniform — a requirement for the FFT-based operations in the Rayleigh–Bénard and quantum modules.

#### 6.1.4 Ghost Cells and Boundary Conditions

Each grid surrounds its real cells with **ghost (halo) layers** used to apply boundary conditions and, in parallel runs, to hold neighbour data. The number of ghost layers depends on the module and its stencil width:

| Module | Ghost layers (per side) | Set by |
|---|---|---|
| Quantum (spectral) | 0 | fixed |
| Rayleigh–Bénard | 1 | fixed |
| Compressible / Convection | 2 – 4 | reconstruction scheme (§6.2.1) |

For the finite-volume modules the ghost count follows the reconstruction order automatically: 2 layers for linear/3rd-order schemes, 3 for 5th-order, 4 for 7th-order.

Boundary conditions are applied by filling the ghost cells. The available wall treatments are:

- **Periodic** — ghost cells wrap to the opposite side (default in the horizontal directions).
- **No-slip / no-penetration** — the field is reflected oddly so it vanishes at the wall (velocity walls).
- **Free-slip / Neumann** — the field is reflected evenly for a zero normal gradient (pressure, free-slip velocity).
- **Dirichlet** — ghost cells are set to a specified boundary value (fixed-temperature walls).

A dedicated interpolation is used to fill vertical ghost cells on non-uniform grids.

#### 6.1.5 Parallel Decomposition

Under MPI (`is_parallel = True`), the grid is split into **slabs along the first axis** — *x* for 1D–3D problems, *w* for 4D Rayleigh–Bénard. Each process owns a contiguous slab plus its ghost layers, and neighbour data is exchanged into those ghost layers each step (Chapter 1, §1.3). The number of real cells along the decomposed axis should be divisible by the number of processes.

#### 6.1.6 Spectral Grids

The Fourier-based modules build a companion wavenumber grid alongside the physical grid:

- **Quantum** — a full spectral grid ($k_x, k_y, k_z$ and $k^2$) for its time-splitting spectral solver.
- **Rayleigh–Bénard** — a spectral grid in the horizontal (periodic) direction, combined with a physical grid in the vertical, matching its FFT-in-*x* / TDMA-in-*y* pressure solve.

### 6.2 Available Methods

The following methods are available in the library. Only those relevant to a given module apply (see the applicability matrix in Chapter 8, §8.7).

#### 6.2.1 Spatial Reconstruction (finite-volume modules)

Selected by `reconstruction`, with an optional `z_smoother` variant:

`linear`, `cweno3`, `weno3`, `weno5`, `teno5`, `weno7`, `teno7`

#### 6.2.2 Convective (inviscid) Flux

Kurganov–Tadmor central-upwind flux family:

- baseline (conservative-variable),
- primitive-variable variant,
- all-speed / low-Mach variant,
- local Lax–Friedrichs variant,
- well-balanced variant (for gravitationally stratified flows).

*These variants are selected in solver code, not through `para.py`.*

#### 6.2.3 Viscous Flux and Derivatives

- Viscous-flux discretization at **2nd, 4th, or 6th order**, selected by `viscous_term_order`.
- Finite-difference derivative operators in 1D–4D, including compact (TDMA-based) variants in 3D and 4D.
- Specialized viscous-flux routines for moist (condensation) physics.

#### 6.2.4 Elliptic (Pressure / Poisson) Solvers

- Preconditioned conjugate gradient (2nd order),
- 4th-order Poisson solver,
- geometric multigrid,
- conjugate-gradient solver,
- spectral + tridiagonal (FFT-in-*x* / TDMA-in-*y*), used by the Rayleigh–Bénard module.

#### 6.2.5 Time Integrators

Selected by `scheme` (finite-volume and Rayleigh–Bénard modules):

- **SSP-RK2** — strong-stability-preserving 2nd-order Runge–Kutta,
- **SSP-RK3** — 3rd-order Runge–Kutta.

The quantum module uses its own time-splitting spectral (TSSP) advance and does not use `scheme`. An IMEX time-stepping hook exists but is reserved/experimental.

---

<a id="ch7"></a>

## Chapter 7 — Available Problems and Sub-Problems

A run is selected by two parameters in `para.py`: `problem` chooses the physics module, and `sub_problem` chooses a specific case within it. This chapter catalogs the cases actually wired into the code.

> **Two kinds of `sub_problem`.** In some modules the `sub_problem` string drives a real code branch (a distinct initial condition or boundary treatment). In others it is only a **descriptive label** — the case is set up elsewhere (in `main.py` or by separate boolean switches) and changing the string has no effect. Each table below marks which is which.

### 7.1 Compressible — Inviscid (`problem = 'euler'`)

| `sub_problem` | Dimensions | Case |
|---|---|---|
| `isentropic` | 2D | Isentropic vortex advection |
| `sod` | 1D, 2D | Sod shock tube (Riemann problem) |
| `sod2` | 2D | Two-dimensional four-quadrant Riemann problem |
| `sine` | 1D | Sinusoidal density/velocity wave |

Each string selects a real initial condition. The 3D Euler solver uses a generic sinusoidal start and has no `sub_problem` branch.

### 7.2 Compressible — Viscous (`problem = 'hydro'`)

| `sub_problem` | Behaviour |
|---|---|
| `channel_flow` | **Functional** — activates extra bookkeeping (saves the streamwise momentum needed to drive the channel). |
| `TG_vortex` | **Label only** — the Taylor–Green initial condition is set directly in the run's `main.py`, not selected by this string. |

For the `hydro` module, `sub_problem` is otherwise a free-text label; only `channel_flow` changes the code path.

### 7.3 Convection (`problem = 'convection'`)

The convection solver has **no `sub_problem` branch**. Its optional physics is enabled by separate boolean switches instead:

| Switch | Companion parameter | Effect |
|---|---|---|
| `shear = True` | `A_f` | Adds sinusoidal shear forcing |
| `heating = True` | `Q_h` | Adds an internal heating source |

The `sub_problem` string in the shipped convection files (e.g. `'shear'`) is decorative and is not read — the boolean `shear` is what turns shear on. Boundary conditions (no-slip walls, fixed wall temperatures) are fixed in code.

> Recall from Chapter 3 that `forcing = True` is also required for a convection run, since gravity is applied through the forcing hook.

### 7.4 Rayleigh–Bénard (`problem = 'rbc_tdma'`)

| `sub_problem` | Dimensions | Case |
|---|---|---|
| `no_slip` | 2D, 3D, 4D | No-slip velocity walls |
| `free_slip` | 2D, 3D | Free-slip velocity walls |

Both strings drive real boundary-condition and initial-condition branches. **`free_slip` is available in 2D and 3D only**; the 4D solver supports `no_slip` alone.

### 7.5 Quantum (`problem = 'quantum'`)

| `sub_problem` | Dimensions | Initial condition |
|---|---|---|
| `random_phase` | 2D, 3D | Band-limited random-phase field (3D uses interpolated phase samples) |
| `random_vortex` | 2D | Loads a pre-generated vortex-tangle wavefunction from file |
| `random_phase_fft` | 3D | FFT-based random-phase field with Hermitian symmetry |

These branches live in the quantum run scripts, so the available values differ by dimension. Independently of `sub_problem`, `potential_type` selects `none` or `harmonic` (Chapter 5).

### 7.6 Experimental / Not Wired Up

The initial-condition files contain setups for several additional problem types that have **no corresponding solver exposed** by the package, so they are not runnable as shipped. They are listed here only so that references to them (in code comments and `para.py` headers) are not mistaken for available cases:

`swe` (shallow-water, `dam_break`), `euler_gravity` / `euler_gravity_WB` (`sod`, `iso_eq`, `explosion`), `euler_mhd` (`orszag`, `sod`), `incomp` (`tgv`), and `FCC_MC`.

### 7.7 Running a Custom Case

Beyond the built-in cases, a run can be started from your own data or continued from a previous run via the `profile` parameter:

- **`profile = 'custom'`** — load initial fields from an HDF5 file in the run's `input/` directory (`<dim>D_input.h5`).
- **`profile = 'continue'`** — resume from a saved snapshot at time `tinit` (Chapter 1, §1.4).

To add a genuinely new case, define its initial condition either in the run's `main.py` or as a new `sub_problem` branch in the corresponding `init_cond` file, then select it from `para.py`.

---

<a id="ch8"></a>

## Chapter 8 — Configuring `para.py`

### 8.0 How `para.py` Is Used

Each run's `main.py` reads every variable from `para.py` into a parameter dictionary passed to the solver and to the data-I/O layer. On startup a copy of `para.py` is written into the output directory as `parameters_<n>.py`, so every result set records the parameters that produced it. Global-quantity and plotting routines are loaded per module and dimension; if a routine is absent for your case, that output is silently skipped (§8.6).

> **Not every block is active.** The shipped files carry leftover parameter blocks from other modules (the compressible file lists `epsilon`/`D`/`Ra`/`Pr`; the convection file lists `Ma`/`Re`). Only the block relevant to the selected `problem` is used — see the per-module tables and the "inert parameter" warnings below.

### 8.1 Simulation Details (all modules)

| Parameter | Example | Allowed values | Effect |
|---|---|---|---|
| `device` | `'GPU'` | `'CPU'`, `'GPU'` | Array backend: NumPy or CuPy |
| `is_parallel` | `False` | bool | Enables MPI domain decomposition |
| `device_rank` | `0` | int | GPU index for a serial GPU run |
| `output_dir` | `'output/test'` | str | Output directory root |
| `dimension` | `2` | `1`, `2`, `3` (`4` for RBC) | Problem dimensionality |
| `problem` | `'hydro'` | `euler`, `hydro`, `convection`, `rbc_tdma`, `quantum` | Physics module |
| `sub_problem` | `'TG_vortex'` | see Chapter 7 | Case selector (some are labels only) |
| `profile` | `'static'` | `static`, `continue`, `custom` | Cold start / restart / load from file |
| `data_type` | `'float64'` | `float64`, `complex128` (quantum) | Array precision/type |

> ⚠️ **`layout_type` is ignored** — every solver fixes its layout internally; the parameter is not read.

### 8.2 Numerical Schemes (finite-volume modules)

| Parameter | Example | Allowed values | Effect |
|---|---|---|---|
| `reconstruction` | `'teno5'` | `linear`, `cweno3`, `weno3`, `weno5`, `teno5`, `weno7`, `teno7` | Spatial reconstruction (also sets ghost count) |
| `z_smoother` | `True` | bool | Z-type smoothness indicators |
| `viscous_term_order` | `2` | `2`, `4`, `6` | Viscous-flux accuracy (`hydro`/`convection`) |
| `scheme` | `'RK3'` | `RK2`, `RK3` | SSP Runge–Kutta integrator |
| `imex` | `False` | bool | Reserved/experimental — leave `False` |

RBC uses `scheme`/`imex` only; quantum uses none of these (fixed spectral scheme).

### 8.3 Time-Stepping

| Parameter | Example | Effect |
|---|---|---|
| `tinit` | `0` | Start time (restart time when `profile='continue'`) |
| `tfinal` | `10` | End time |
| `dt` | `2.5e-3` | Base time step (upper bound under CFL) |
| `cfl_condition` | `False` | Recompute `dt` from a CFL limit each step |
| `cfl_cons` | `0.5` | CFL safety factor |

> Quantum has no `cfl_condition`/`cfl_cons` (fixed `dt`).

### 8.4 Grid and Domain

| Parameter | Example | Effect |
|---|---|---|
| `L_min` | `[-np.pi, -np.pi]` | Lower bounds (length = `dimension`) |
| `L_max` | `[np.pi, np.pi]` | Upper bounds (length = `dimension`) |
| `N` | `[512, 512]` | Grid points per axis |
| `spacing_type` | `'uniform'` | `uniform` or `non-uniform` |
| `beta` | `1.2` | Clustering strength for `non-uniform` |

> ⚠️ **Non-uniform clustering acts in the *y*-direction only**; *x* is always uniform. Higher `beta` ⇒ stronger clustering toward the top/bottom walls. Ignored by the quantum module.

### 8.5 Physical Control Parameters (per module)

#### 8.5.1 Compressible — `hydro` / `euler`

| Parameter | Effect |
|---|---|
| `gamma` | Ratio of specific heats |
| `Ma` | Mach number (R_gas = 1/(γ·Ma²)) |
| `Re` | Reynolds number (μ = 1/Re; `hydro` only) |
| `Pr` | Prandtl number |
| `mu_varying` | Temperature-dependent viscosity toggle |
| `forcing`, `forced_vrms`, `zeta` | Stochastic turbulence forcing and its target RMS / compressibility |
| `rad_cooling`, `alpha_cooling` | Radiative-cooling source and its rate |

`euler` uses `gamma` only.

#### 8.5.2 Convection — `convection`

| Parameter | Effect |
|---|---|
| `gamma` | Ratio of specific heats |
| `epsilon` | Superadiabaticity (thermal driving) |
| `D` | Dissipation number (stratification depth) |
| `Ra`, `Pr` | Rayleigh, Prandtl numbers |
| `forcing` | **Required — enables gravity** (see warning) |
| `shear`, `A_f` | Shear forcing and its amplitude |
| `heating`, `Q_h` | Internal heating and its rate |

> ⚠️ **`forcing = True` is mandatory** — gravity is applied through the forcing hook. With `forcing = False` there is no buoyancy and the run is not convection.
>
> ⚠️ **`Ma` and `Re` are inert here** — they are read by the inherited constructor but immediately overwritten by values derived from ε / D / Ra / Pr.

#### 8.5.3 Rayleigh–Bénard — `rbc_tdma`

| Parameter | Effect |
|---|---|
| `Ra`, `Pr` | Rayleigh, Prandtl numbers (ν = √(Pr/Ra), κ = 1/√(Ra·Pr)) |
| `shear`, `A_f` | Optional mean shear force and amplitude |

#### 8.5.4 Quantum — `quantum`

| Parameter | Effect |
|---|---|
| `potential_type` | `none` or `harmonic` trap |
| `alpha` | Dispersion (Laplacian) coefficient |
| `g0` | Nonlinearity (0 ⇒ linear Schrödinger) |
| `forcing`, `Q`, `kp`, `kr` | Spectral-shell forcing and its band |
| `hyper_dissipation`, `nu_hyper`, `kh` | High-k damping |
| `hypo_dissipation`, `nu_hypo`, `order_hypo`, `k_hypo` | Low-k damping |
| `condensate_sink`, `nu_condensate` | Condensate removal (**3D only**) |

### 8.6 Output and Saving (all modules)

| Parameter | Effect |
|---|---|
| `print_out_terminal` | Print per-step status line |
| `save_fields`, `save_fields_dt` | Write field snapshots and their interval |
| `save_global_quantities`, `save_global_quantities_dt` | Accumulate integral diagnostics and sampling interval |
| `save_global_file` | Interval at which globals are flushed to disk |
| `save_plots`, `save_plots_tinit`, `save_plots_dt` | Render PNG frames (+ movie) and their timing |
| `save_plot_fields`, `N_dump_fields` | Dump reduced fields for offline plotting |
| `save_mean_quantities` | Accumulate horizontally-averaged mean profiles |

> ⚠️ **Plotting is module- and dimension-gated.** A plot routine must exist for your `problem`/`dimension` or `save_plots` has no effect. As shipped: convection has 1D plots only, quantum 3D only, RBC 2D only, hydro 2D/3D, euler 1D/2D.

### 8.7 Module Applicability Matrix

| Group | euler | hydro | convection | rbc_tdma | quantum |
|---|:--:|:--:|:--:|:--:|:--:|
| `reconstruction`, `z_smoother` | ✓ | ✓ | ✓ | — | — |
| `viscous_term_order` | — | ✓ | ✓ | — | — |
| `scheme` (RK2/RK3) | ✓ | ✓ | ✓ | ✓ | — |
| `cfl_condition`, `cfl_cons` | ✓ | ✓ | ✓ | ✓ | — |
| `spacing_type`, `beta` | ✓ | ✓ | ✓ | ✓ | — |
| `gamma`, `Ma`, `Re`, `Pr` | `gamma` | ✓ | inherited but inert | — | — |
| `epsilon`, `D`, `Ra`, `Pr` | — | — | ✓ | `Ra`, `Pr` | — |
| `forcing` (+ shear/heating) | stochastic | stochastic | **gravity gate** | shear only | spectral |
| `g0`, `alpha`, `potential_type`, hyper/hypo | — | — | — | — | ✓ |

---

<a id="ch9"></a>

## Chapter 9 — Worked Example: 512 × 512 Taylor–Green Vortex on GPU

This example runs the classic Taylor–Green vortex as a 2D compressible Navier–Stokes problem on the GPU, and walks through the output it produces. It exercises the compressible module (Chapter 2) end-to-end: configuration, execution, and interpretation of the saved diagnostics.

### Part 1 — Running the Simulation

**Step 1.** Open `compressible/2d/para.py`.

**Step 2.** Set the following (leave all other parameters at their defaults):

```python
# --- Simulation details ---
device      = 'GPU'
is_parallel = False
device_rank = 0

dimension   = 2
problem     = 'hydro'
sub_problem = 'TG_vortex'
profile     = 'static'

reconstruction = 'teno5'
scheme         = 'RK3'
data_type      = 'float64'

# --- Grid / domain ---
L_min = [-np.pi, -np.pi]
L_max = [ np.pi,  np.pi]
N     = [512, 512]

# --- Time stepping ---
tinit  = 0
tfinal = 10
dt     = 2.5e-3
cfl_condition = False

# --- Output ---
save_fields    = True
save_fields_dt = 2.5

# --- Physics ---
gamma = 1.4
Ma    = 1.25
Re    = 1600
```

**Step 3.** Run the solver from the example directory:

```bash
cd compressible/2d
```

```bash
python main.py
```

The driver builds a compressible Navier–Stokes problem with the analytic Taylor–Green initial condition (a periodic array of counter-rotating vortices) and advances it with 5th-order TENO reconstruction and third-order Runge–Kutta until `tfinal`.

### Part 2 — Results

#### 9.1 How the Output Is Saved

With the configuration above, two kinds of HDF5 output appear in `output/test/`.

**Field snapshots**, written every `save_fields_dt = 2.5` time units:

```
output/test/fields/2D_0.00.h5
output/test/fields/2D_2.50.h5
output/test/fields/2D_5.00.h5
output/test/fields/2D_7.50.h5
output/test/fields/2D_10.00.h5
```

Each snapshot contains the flow fields `rho`, `p`, `ux`, `uy` (2D arrays), the coordinate arrays `x`, `y`, and the time `t`.

**Global-quantities file** (`glob_0.00.h5`) records integrated diagnostics sampled every `save_global_quantities_dt`:

| Dataset | Quantity |
|---|---|
| `M_T` | Total mass |
| `I_T` | Total internal energy |
| `E_T` | Total energy |
| `K_T` | Total kinetic energy |
| `V_rms` | Root-mean-square velocity |
| `Re` | Reynolds number |
| `Ma_max` | Maximum Mach number |
| `Ma_avg` | Volume-averaged Mach number |
| `t` | Sample times |

#### 9.2 Final Velocity Field

![Final velocity field at t = 10](https://github.com/user-attachments/assets/d58e7a99-3b5a-4fdc-9421-13d4ed9909bc)

*Velocity magnitude at $t = 10$, produced from `2D_10.00.h5`, with the velocity vectors overlaid in white:*

$$\lVert \mathbf{u} \rVert = \sqrt{u_x^2 + u_y^2}$$

The velocity magnitude $\lVert\mathbf{u}\rVert$ at the final time retains the fourfold symmetry of the initial condition. High-speed shear layers separate the vortices, while vortex cores and stagnation points approach zero velocity; overlaid vectors trace the circulation around each core.

#### 9.3 Energy Budget vs. Time

![Kinetic, internal and total energy vs time](https://github.com/user-attachments/assets/6cdd2d60-b4b3-40b6-97ee-3c329837c033)

*Total kinetic (`K_T`), internal (`I_T`) and total (`E_T`) energy vs time, from
`glob_0.00.h5`.*

Total energy `E_T` stays essentially constant, confirming the scheme's conservative behavior. Kinetic energy `K_T` decreases monotonically (from ≈ 9.85 to ≈ 5.5–6.0 by $t = 10$) as internal energy `I_T` rises correspondingly — the signature of viscous dissipation converting kinetic energy into heat.

Superimposed oscillations reflect acoustic dynamics inherent to compressibility: kinetic and internal energy exchange through pressure work while viscosity drains the total kinetic energy. Local kinetic-energy peaks occur near $t \approx 2.5,\ 5.8,\ 9$, with troughs near $t \approx 4.5,\ 7.7$.

#### 9.4 Kinetic Energy Decay

![Total kinetic energy vs time](https://github.com/user-attachments/assets/2150fcd2-6d4a-4c42-8a62-af152b1e2003)

*Total kinetic energy `K_T` vs time (a zoom of the blue curve above).*

The kinetic-energy decay curve isolates the dissipation trend from the acoustic oscillations, giving the effective decay rate of the vortex array at the chosen Reynolds number — the standard quantity used to validate a compressible solver against Taylor–Green reference data.

#### 9.5 Mass Conservation

![Total mass vs time](https://github.com/user-attachments/assets/d1cbdb05-2fd5-4350-8cba-64a225061042)

*Total mass `M_T` vs time, from `glob_0.00.h5`.*

Total mass holds at ≈ 39.4784176, varying by only $\sim 10^{-11}$ over the entire run — conservation to machine precision. The faint downward drift is floating-point round-off, not physical mass loss, confirming the finite-volume formulation's discrete mass conservation.
