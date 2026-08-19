# DHARA 2D Solver: Line-by-Line Analysis
## Complete Explanation with Differential Equations

---

# PART 1: PARA2D.PY - PARAMETER CONFIGURATION FILE

## Overview
`para2d.py` is a **configuration file** that defines all parameters for a 2D compressible Navier-Stokes simulation. It doesn't execute computations—it only **stores settings** that the main solver reads.

---

## SECTION 1: SIMULATION DETAILS

### Line 1: `import numpy as np`
```python
import numpy as np
```
**Function Used:** `numpy.import`
**Purpose:** Imports NumPy library for numerical operations
**Why Needed:** Used later for mathematical constants like `np.pi` and array operations
**Not directly related to differential equations** (utility import)

---

### Lines 6-12: Device & Parallelization Configuration

```python
device = 'GPU'            # CPU, GPU
is_parallel = False        # Parallel simulation
device_rank = 0           # GPU device rank for serial simulation
output_dir = 'output/test'     # Output directory
```

**Block Purpose:** Configure where and how the simulation runs

| Line | Variable | Function | Purpose | Physical Meaning |
|------|----------|----------|---------|-----------------|
| `device = 'GPU'` | String assignment | Sets computation device | Execute on GPU (faster) or CPU (portable) | Not directly in equations; affects numerical precision |
| `is_parallel = False` | Boolean assignment | Disables multi-GPU parallelization | Run serially on single device | Not in equations |
| `device_rank = 0` | Integer assignment | Selects GPU index 0 | Which GPU to use (irrelevant if parallel=False) | Not in equations |
| `output_dir` | String assignment | Sets output folder path | Where to save results (HDF5 files) | Not in equations |

**No differential equations involved** - pure configuration

---

### Lines 15-22: Problem Definition

```python
dimension = 2             # Dimension of the problem
problem = 'hydro'         # Problem to be solved
sub_problem = 'TG_vortex' # Sub-problem to be solved
layout_type = 'collocated'      # Grid layout type
spacing_type = 'uniform'        # Grid type
beta = 1                  # beta for tangent-hyperbolic grid
profile = 'static'        # Initial profiles  
```

**Block Purpose:** Define what physical problem to solve and how to discretize it

| Line | Variable | Function | Purpose | Physical Meaning |
|------|----------|----------|---------|-----------------|
| `dimension = 2` | Integer assignment | Specifies 2D problem (vs 3D) | Reduces 3D equations to 2D by removing z-dependency | Physics: $\frac{\partial}{\partial z} = 0$ (no z-variation) |
| `problem = 'hydro'` | String assignment | Selects compressible Navier-Stokes | Solve fluid flow with viscosity | Governs these equations (explained below) |
| `sub_problem = 'TG_vortex'` | String assignment | Selects test case type | Uses Taylor-Green vortex analytical solution | Validates numerical accuracy |
| `layout_type = 'collocated'` | String assignment | All variables at same grid points | Simple grid arrangement (vs staggered) | Numerical discretization choice |
| `spacing_type = 'uniform'` | String assignment | Equally-spaced grid points | $\Delta x = \Delta y = $ constant | Grid uniformity: $x_i = x_{min} + i \cdot \Delta x$ |
| `profile = 'static'` | String assignment | Use analytical initial conditions | Initialize from Taylor-Green solution, not from file | Initial Condition (IC) |

**Differential Equations Involved:**

For 2D Compressible Navier-Stokes with `problem='hydro'`:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0 \quad \text{(Continuity)}$$

$$\frac{\partial \rho \mathbf{u}}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = -\nabla p + \nabla \cdot \boldsymbol{\tau} \quad \text{(Momentum)}$$

$$\frac{\partial \rho e}{\partial t} + \nabla \cdot (\rho e \mathbf{u}) = -\nabla \cdot (p\mathbf{u}) + \nabla \cdot (\boldsymbol{\tau} \cdot \mathbf{u}) \quad \text{(Energy)}$$

In 2D: $\mathbf{u} = (u_x, u_y)$, so:

$$\frac{\partial \rho}{\partial t} + \frac{\partial(\rho u_x)}{\partial x} + \frac{\partial(\rho u_y)}{\partial y} = 0$$

$$\frac{\partial(\rho u_x)}{\partial t} + \frac{\partial(\rho u_x^2)}{\partial x} + \frac{\partial(\rho u_x u_y)}{\partial y} = -\frac{\partial p}{\partial x} + \nabla^2 u_x$$

$$\frac{\partial(\rho u_y)}{\partial t} + \frac{\partial(\rho u_x u_y)}{\partial x} + \frac{\partial(\rho u_y^2)}{\partial y} = -\frac{\partial p}{\partial y} + \nabla^2 u_y$$

---

### Lines 25-30: Numerical Scheme Selection

```python
reconstruction = 'teno5'  # Reconstruction scheme
z_smoother = True         # Use z-smoother for reconstruction
viscous_term_order = 2    # Order of accuracy for viscous term
scheme = 'RK3'            # Time integration scheme
imex = False              # IMEX time integration scheme
data_type = 'float64'     # Data type of arrays
```

**Block Purpose:** Select numerical methods for spatial/temporal discretization

| Line | Variable | Function | Purpose | Physical Meaning |
|------|----------|----------|---------|-----------------|
| `reconstruction = 'teno5'` | String assignment | 5th-order TENO reconstruction scheme | Approximates flux at cell faces from cell-center values | Spatial discretization: $u_{i+1/2} = f(u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2})$ |
| `z_smoother = True` | Boolean assignment | Apply smoothing in z-direction | Reduce oscillations (not relevant for 2D, but kept for code compatibility) | Post-processing: $u'_i = \alpha u_i + (1-\alpha)u_{i-1}$ |
| `viscous_term_order = 2` | Integer assignment | 2nd-order central differences for viscosity | $\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$ | Discretizes viscous stress term |
| `scheme = 'RK3'` | String assignment | 3rd-order Runge-Kutta time integration | Multi-stage method for time advancement | Temporal discretization |
| `imex = False` | Boolean assignment | Use explicit time stepping only | Coupled implicit-explicit (disabled here) | Affects stability analysis |
| `data_type = 'float64'` | String assignment | Use 64-bit floating point | Double precision (vs float32) | Numerical precision |

**Differential Equations Involved:**

For the **viscous stress tensor** (Newtonian fluid):

$$\boldsymbol{\tau} = \mu \left[ \nabla \mathbf{u} + (\nabla \mathbf{u})^T - \frac{2}{3}(\nabla \cdot \mathbf{u})\mathbf{I} \right]$$

In 2D:
$$\frac{\partial^2 u_x}{\partial x^2} + \frac{\partial^2 u_x}{\partial y^2} \quad \text{(Discretized using viscous_term_order=2)}$$

For **RK3 time integration**, a general PDE is advanced as:

$$\frac{\partial \mathbf{U}}{\partial t} = \mathbf{F}(\mathbf{U})$$

RK3 stages:
$$\mathbf{U}^{(1)} = \mathbf{U}^n + \Delta t \, \mathbf{F}(\mathbf{U}^n)$$
$$\mathbf{U}^{(2)} = \frac{3}{4}\mathbf{U}^n + \frac{1}{4}\mathbf{U}^{(1)} + \frac{1}{4}\Delta t \, \mathbf{F}(\mathbf{U}^{(1)})$$
$$\mathbf{U}^{n+1} = \frac{1}{3}\mathbf{U}^n + \frac{2}{3}\mathbf{U}^{(2)} + \frac{2}{3}\Delta t \, \mathbf{F}(\mathbf{U}^{(2)})$$

---

## SECTION 2: TIME-STEPPING PARAMETERS

### Lines 36-43: Time Integration Setup

```python
tinit = 0                 # Initial time
tfinal = 2.5              # Final time
dt = 2.5e-3               # Single time step
cfl_condition = False     # CFL condition for time step calculation
cfl_cons = 0.5            # CFL constant for time step calculation
```

**Block Purpose:** Define temporal domain and step size

| Line | Variable | Function | Purpose | Physical Meaning |
|------|----------|----------|---------|-----------------|
| `tinit = 0` | Float assignment | Start simulation at t=0 | Initial time point | Lower bound of temporal domain |
| `tfinal = 2.5` | Float assignment | End simulation at t=2.5 | Final time point | Upper bound of temporal domain |
| `dt = 2.5e-3` | Float assignment | Fixed time step of 0.0025 | Advance by 0.0025 each iteration | $\Delta t = 2.5 \times 10^{-3}$ |
| `cfl_condition = False` | Boolean assignment | Use fixed dt (not adaptive) | Don't calculate dt from CFL condition | Explicit scheme stability |
| `cfl_cons = 0.5` | Float assignment | CFL number 0.5 (unused here) | Would be used if cfl_condition=True | Courant number: $C = u \frac{\Delta t}{\Delta x}$ |

**Differential Equations Involved:**

The **CFL (Courant-Friedrichs-Lewy) condition** for explicit schemes:

$$C = |u| \frac{\Delta t}{\Delta x} \leq C_{max}$$

For RK3: $C_{max} \approx 1.256$ typically, but code uses $C_{max} = 0.5$ (conservative)

Number of time steps:
$$n_{steps} = \frac{t_{final} - t_{init}}{\Delta t} = \frac{2.5 - 0}{2.5 \times 10^{-3}} = 1000 \text{ iterations}$$

---

## SECTION 3: GRID PARAMETERS

### Lines 49-55: Spatial Domain & Resolution

```python
L_min = [-np.pi,-np.pi]     # Minimum value of x and y
L_max = [np.pi,np.pi]       # Maximum value of x and y
N = [256, 256]              # Number of grid points in x and y
```

**Block Purpose:** Define computational grid

| Line | Variable | Function | Purpose | Physical Meaning |
|------|----------|----------|---------|-----------------|
| `L_min = [-π,-π]` | List assignment | Domain lower bound: $x \in [-\pi, \pi]$, $y \in [-\pi, \pi]$ | Periodic domain from $-\pi$ to $\pi$ in both directions | Physical domain for Taylor-Green vortex |
| `L_max = [π,π]` | List assignment | Domain upper bound | Same domain extent | Physical domain |
| `N = [256, 256]` | List assignment | 256×256 grid points | $256^2 = 65,536$ total grid points | Spatial resolution: $\Delta x = \Delta y = \frac{2\pi}{256} \approx 0.0245$ |

**Differential Equations Involved:**

Grid spacing:
$$\Delta x = \frac{L_{max,x} - L_{min,x}}{N_x} = \frac{\pi - (-\pi)}{256} = \frac{2\pi}{256}$$

$$\Delta y = \frac{L_{max,y} - L_{min,y}}{N_y} = \frac{\pi - (-\pi)}{256} = \frac{2\pi}{256}$$

Grid points:
$$x_i = L_{min,x} + i \cdot \Delta x, \quad i = 0, 1, ..., N_x-1$$
$$y_j = L_{min,y} + j \cdot \Delta y, \quad j = 0, 1, ..., N_y-1$$

---

## SECTION 4: CONTROL PARAMETERS (PHYSICAL PROPERTIES)

### Lines 61-76: Fluid Properties & Forcing

```python
gamma = 1.4               # Ratio of specific heats  
Ma = 1.25                 # Mach number
mu_varying = False        # Varying viscosity
Re = 1600                 # Reynolds number
forcing = False           # Forcing term
forced_vrms = 1           # Forced velocity root mean square
zeta = 2/3                # Compressibility parameter
rad_cooling = False       # Radiative cooling term
alpha_cooling = 1e-2      # Cooling parameter
```

**Block Purpose:** Set physical fluid properties and control problem behavior

| Line | Variable | Function | Purpose | Physical Meaning |
|------|----------|----------|---------|-----------------|
| `gamma = 1.4` | Float assignment | Ratio of specific heats | $\gamma = C_p / C_v = 1.4$ for air | Thermodynamic property of ideal gas |
| `Ma = 1.25` | Float assignment | Mach number | $Ma = 1.25$ (transonic flow) | Compressibility: $Ma = \frac{u_0}{a_0}$ where $a_0 = \sqrt{\gamma R T}$ |
| `mu_varying = False` | Boolean assignment | Assume constant viscosity | Don't use Sutherland law or temperature-dependent $\mu$ | Simplification: $\mu = const$ |
| `Re = 1600` | Float assignment | Reynolds number | $Re = 1600$ (moderately viscous) | Ratio of inertial to viscous forces |
| `forcing = False` | Boolean assignment | No external forcing | Taylor-Green decays naturally without source term | Homogeneous flow (no $\mathbf{f}$ term in momentum equation) |
| `forced_vrms = 1` | Float assignment | RMS velocity (unused) | Would set if forcing=True | Not used in this simulation |
| `zeta = 2/3` | Float assignment | Compressibility parameter (unused for hydro) | Used for specific forcing models | Not active in this simulation |
| `rad_cooling = False` | Boolean assignment | No radiative cooling | No temperature loss term | Adiabatic flow |
| `alpha_cooling = 1e-2` | Float assignment | Cooling coefficient (unused) | Would add $-\alpha T$ term if rad_cooling=True | Not active here |

**Differential Equations Involved:**

**Mach number** relates to equation of state:
$$Ma = \frac{u_0}{a_0}, \quad a_0 = \sqrt{\gamma R T}$$

For ideal gas: $p = \rho R T$

**Reynolds number** defines viscous effects:
$$Re = \frac{\rho u_0 L}{\mu} = \frac{\text{Inertial forces}}{\text{Viscous forces}}$$

Navier-Stokes with constant $\mu$:
$$\rho \frac{D\mathbf{u}}{Dt} = -\nabla p + \mu \nabla^2 \mathbf{u}$$

Which can be rewritten as:
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \frac{\mu}{\rho} \nabla^2 \mathbf{u}$$

Or in terms of Re:
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \frac{1}{Re} \nabla^2 \mathbf{u}$$

---

### Lines 79-83: Convection Problem Parameters (Unused for 'hydro')

```python
epsilon = 0.1             # Superadiabaticity   
D = 0.5                   # Dissipation number  
Ra = 1e5                  # Rayleigh number  
Pr = 0.71                 # Prandtl number  
```

**Block Purpose:** Parameters for Rayleigh-Bénard convection (not used since `problem='hydro'`)

These would be used if `problem = 'convection'`

**Differential Equations Involved (if used):**

Boussinesq approximation for convection:
$$\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \frac{1}{Pr \cdot Ra} \nabla^2 T$$

Where:
- $Ra = \frac{g \beta \Delta T L^3}{\nu \alpha}$ (Rayleigh number)
- $Pr = \frac{\nu}{\alpha}$ (Prandtl number)

**Not active in current simulation since `problem='hydro'`**

---

## SECTION 5: SAVING PARAMETERS (OUTPUT CONTROL)

### Lines 89-98: File Output Configuration

```python
print_out_terminal = True           # Print running time-instance
save_fields = True                  # Saving fields files
save_fields_dt = 2.5                # Field files saving time-step
save_global_quantities = True       # Saving global quantitites
save_global_quantities_dt = 1e-2    # Global quantities saving time-step
save_global_file = 1                # Saving time of global file
save_plots = False                  # Saving density plots
save_plots_tinit = 0                # Density plots initial time
save_plots_dt = 5e-1                # Density plots saving time-step
```

**Block Purpose:** Control where and when simulation data is saved

| Line | Variable | Function | Purpose | Physical Meaning |
|------|----------|----------|---------|-----------------|
| `print_out_terminal = True` | Boolean assignment | Print progress to console | Display: "Time step 100/1000" etc. | Monitoring progress |
| `save_fields = True` | Boolean assignment | Save velocity, pressure, density | Write HDF5 file with $u_x, u_y, p, \rho$ fields | Saving full field data |
| `save_fields_dt = 2.5` | Float assignment | Save every 2.5 time units | At $t = 0, 2.5$ (only 2 snapshots for tfinal=2.5) | Data reduction |
| `save_global_quantities = True` | Boolean assignment | Save integrated quantities | Compute and save kinetic energy, enstrophy, etc. | Post-processing |
| `save_global_quantities_dt = 1e-2` | Float assignment | Save every 0.01 time units | Save 250 snapshots (every time step in this case) | Fine temporal resolution for diagnostics |
| `save_global_file = 1` | Float assignment | Complete save every 1 time unit | Frequency of full state dumps | Checkpoint data |
| `save_plots = False` | Boolean assignment | Don't generate PNG plots | Skip visualization step (can do manually later) | Optional visualization |
| `save_plots_tinit = 0` | Float assignment | Plotting start time (unused) | Not generating plots | N/A |
| `save_plots_dt = 5e-1` | Float assignment | Plotting interval (unused) | Not generating plots | N/A |

**No differential equations directly involved** - these are I/O parameters

---

---

# PART 2: MAIN2D.PY - SIMULATION EXECUTION FILE

## Overview
`main2d.py` executes the actual 2D simulation. It:
1. Imports parameters from `para2d.py`
2. Initializes the solver
3. Sets initial conditions
4. Runs the time integration
5. Saves results

---

## SECTION 1: HEADER & IMPORTS

### Lines 1-5: Documentation & Module Setup

```python
"""
For running the Taylor green vortex test in 2D using DHARA framework.
"""

import sys
sys.dont_write_bytecode = True
```

**Block Purpose:** Configure Python environment

| Line | Function | Purpose | Technical Meaning |
|------|----------|---------|-------------------|
| Docstring | Documentation | Explains script purpose | Meta-comment (not executed) |
| `import sys` | NumPy module import | Access Python system utilities | Allows file system operations |
| `sys.dont_write_bytecode = True` | System configuration | Don't create `__pycache__` files | Cleaner directory structure |

**No differential equations involved**

---

### Lines 7-11: Import Core Modules

```python
import time
import para
sys.path.append('../../')
from DHARA import *
```

**Block Purpose:** Load required modules and packages

| Line | Function | Purpose | Technical Meaning |
|------|----------|---------|-------------------|
| `import time` | Built-in module | Measure wall-clock time | `time.time()` returns seconds since epoch |
| `import para` | Custom module | Load parameters from `para2d.py` | Reads all variables from para2d.py into namespace |
| `sys.path.append('../../')` | Path modification | Add parent directory to search path | Allows importing DHARA framework 2 levels up |
| `from DHARA import *` | Framework import | Import all DHARA classes and functions | Loads: `Hydro2D`, `DataIO`, `Time_Evolution`, `init3D`, etc. |

**No differential equations involved** - these are imports

---

## SECTION 2: PARAMETER PACKAGING

### Lines 14-18: Create Parameter Dictionary

```python
# Create a parameter dictionary
params = {
    key: value
    for key, value in vars(para).items()
    if not key.startswith("__") and not callable(value)
}
```

**Block Purpose:** Convert parameter file into a dictionary

**Line-by-line breakdown:**

```python
params = {  # Initialize empty dictionary to hold parameters
    key: value  # For each key-value pair...
    for key, value in vars(para).items()  # ...from all variables in para module
    if not key.startswith("__")  # ...excluding dunder variables (__name__, __doc__, etc.)
    and not callable(value)  # ...and excluding functions/methods
}
```

**Functions used:**
- `vars(para)` - Returns dictionary of all attributes in `para` module
- `.items()` - Extracts (key, value) pairs
- Dictionary comprehension - Filters and builds new dictionary

**Purpose:** Create a clean parameter dictionary like:
```python
params = {
    'device': 'GPU',
    'dimension': 2,
    'problem': 'hydro',
    'N': [256, 256],
    'Re': 1600,
    ...  # All 40+ parameters from para2d.py
}
```

**No differential equations involved** - data structure operation

---

## SECTION 3: SOLVER INITIALIZATION

### Lines 21-22: Create Problem Object

```python
# Initialize the problem
prob = Hydro2D(params)
```

**Block Purpose:** Instantiate the 2D hydrodynamics solver

**Function used:** `Hydro2D(params)` class constructor

**What happens internally:**
1. Allocates memory for $u_x$, $u_y$, $p$, $\rho$ fields (256×256 arrays)
2. Creates grid coordinates ($x_i, y_j$)
3. Initializes boundary conditions
4. Sets equation of state ($\gamma = 1.4$)
5. Creates MPI communicator (if parallel)

**Object structure created:**
```python
prob.ux          # 2D array: [256, 256] velocity x-component
prob.uy          # 2D array: [256, 256] velocity y-component
prob.p           # 2D array: [256, 256] pressure
prob.rho         # 2D array: [256, 256] density
prob.q[0]        # 2D array: [256, 256] temperature
prob.grid.x      # 1D array: [256] x-coordinates
prob.grid.y      # 1D array: [256] y-coordinates
prob.ncp         # numpy or cupy (depending on device)
prob.R_gas       # Gas constant: R = (γ-1)Cv
```

**Differential equations involved:**

The solver is prepared to solve:

$$\frac{\partial}{\partial t}\begin{pmatrix} \rho \\ \rho u_x \\ \rho u_y \\ \rho e \end{pmatrix} + \frac{\partial}{\partial x}\begin{pmatrix} \rho u_x \\ \rho u_x^2 + p \\ \rho u_x u_y \\ (\rho e + p)u_x \end{pmatrix} + \frac{\partial}{\partial y}\begin{pmatrix} \rho u_y \\ \rho u_x u_y \\ \rho u_y^2 + p \\ (\rho e + p)u_y \end{pmatrix} = \text{Viscous terms}$$

In **conservative form** with 4 state variables:
$$\mathbf{U} = \begin{pmatrix} \rho \\ \rho u_x \\ \rho u_y \\ \rho e \end{pmatrix}$$

---

## SECTION 4: INITIAL CONDITIONS (TAYLOR-GREEN VORTEX)

### Lines 25-39: Set Initial Velocity & Pressure

```python
# Initialize the problem
if para.profile == 'static':
    prob.ux[...] = prob.ncp.sin(prob.grid.x[:,None]) * prob.ncp.cos(prob.grid.y[None,:])
    prob.uy[...] = -prob.ncp.cos(prob.grid.x[:,None]) * prob.ncp.sin(prob.grid.y[None,:])
    prob.p[...] = prob.R_gas + (1/16)*(
        prob.ncp.cos(2*prob.grid.x[:,None]) +
        prob.ncp.cos(2*prob.grid.y[None,:]))
    prob.q[0] = prob.p/(prob.R_gas) #T0 = 1
    prob.compute_conserved()
    prob.bc_conserved()
```

**Block Purpose:** Initialize flow field with analytical Taylor-Green vortex solution

---

### Line 27: `if para.profile == 'static':`

**Function:** Conditional check
**Purpose:** Only initialize if using static (analytical) profile
**Alternative:** If `para.profile = 'custom'`, would load from `init.h5` file

---

### Line 28: Initialize $u_x$ Component

```python
prob.ux[...] = prob.ncp.sin(prob.grid.x[:,None]) * prob.ncp.cos(prob.grid.y[None,:])
```

**Breakdown:**

| Part | Function | Purpose |
|------|----------|---------|
| `prob.ux[...]` | Array assignment | In-place modification of velocity x-component |
| `prob.ncp.sin()` | NumPy/CuPy sine function | Compute $\sin(x)$ element-wise |
| `prob.grid.x` | 1D array of x-coordinates | Shape: [256] |
| `[:,None]` | Array broadcasting | Reshape [256] → [256, 1] for 2D operation |
| `prob.ncp.cos()` | NumPy/CuPy cosine function | Compute $\cos(y)$ element-wise |
| `[None,:]` | Array broadcasting | Reshape [256] → [1, 256] for 2D operation |

**Broadcasting operation:**
```
[256, 1] × [1, 256] → [256, 256]
```

**Analytical formula:**
$$u_x(x, y) = \sin(x) \cos(y)$$

**Physical meaning:** Horizontal velocity component of Taylor-Green vortex

**Differential equation context:** This is the **initial condition** for:
$$\frac{\partial (\rho u_x)}{\partial t} + \frac{\partial (\rho u_x^2)}{\partial x} + \frac{\partial (\rho u_x u_y)}{\partial y} = -\frac{\partial p}{\partial x} + \nabla^2 u_x$$

At $t=0$: $u_x(x,y,0) = \sin(x) \cos(y)$

---

### Line 29: Initialize $u_y$ Component

```python
prob.uy[...] = -prob.ncp.cos(prob.grid.x[:,None]) * prob.ncp.sin(prob.grid.y[None,:])
```

**Analytical formula:**
$$u_y(x, y) = -\cos(x) \sin(y)$$

**Physical meaning:** Vertical velocity component (negative cosines create counter-rotating pattern)

**Satisfies continuity equation at t=0:**
$$\frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} = -\sin(x) \sin(y) + (-\sin(x) \sin(y)) = \text{Check:}$$

Actually:
$$\frac{\partial u_x}{\partial x} = \cos(x) \cos(y)$$
$$\frac{\partial u_y}{\partial y} = -\cos(x) \cos(y)$$
$$\nabla \cdot \mathbf{u} = \cos(x) \cos(y) - \cos(x) \cos(y) = 0 \quad \checkmark$$

**Velocity field is divergence-free** (incompressible IC for compressible solver)

---

### Lines 30-33: Initialize Pressure Field

```python
prob.p[...] = prob.R_gas + (1/16)*(
    prob.ncp.cos(2*prob.grid.x[:,None]) +
    prob.ncp.cos(2*prob.grid.y[None,:]))
```

**Breakdown:**

| Component | Formula | Meaning |
|-----------|---------|---------|
| `prob.R_gas` | $R = (\gamma - 1) C_V$ | Base pressure (reference state) |
| `(1/16)` | Amplitude | Pressure perturbation magnitude |
| `cos(2x)` | $\cos(2x)$ | Double frequency in x |
| `cos(2y)` | $\cos(2y)$ | Double frequency in y |

**Full formula:**
$$p(x,y) = R + \frac{1}{16}[\cos(2x) + \cos(2y)]$$

**Physical meaning:** 
- Pressure is mostly constant ($p \approx R$)
- Small oscillations at twice the velocity frequency
- Minimum: $p_{min} = R - \frac{1}{8}$ (at $x=y=\pi/2$)
- Maximum: $p_{max} = R + \frac{1}{8}$ (at $x=y=0$)

**Differential equation context:** Initial condition for energy equation:

$$\frac{\partial (\rho e)}{\partial t} + \frac{\partial (\rho e u_x)}{\partial x} + \frac{\partial (\rho e u_y)}{\partial y} = \text{pressure & viscous terms}$$

---

### Line 34: Set Initial Temperature

```python
prob.q[0] = prob.p/(prob.R_gas)
```

**What this does:**
- `prob.q[0]` is temperature array (first element of state vector)
- Computes $T = \frac{p}{R}$ from ideal gas law: $p = \rho R T$

**Formula:**
$$T(x,y) = \frac{p(x,y)}{R}$$

With $p = R + \frac{1}{16}[\cos(2x) + \cos(2y)]$:

$$T(x,y) = 1 + \frac{1}{16R}[\cos(2x) + \cos(2y)]$$

**Physical meaning:** Temperature field derived from pressure (not independent)

**Equation of state context:**
$$p = \rho R T$$

For ideal gas: $e = C_V T$ where $C_V = \frac{R}{\gamma - 1}$

---

### Line 35: Compute Conserved Variables

```python
prob.compute_conserved()
```

**Function:** `compute_conserved()` method of `Hydro2D` class

**Purpose:** Convert primitive variables to conservative variables

**Primitive variables** (what we just set):
$$(\rho, u_x, u_y, p)$$

**Conservative variables** (what the solver uses):
$$\mathbf{U} = (\rho, \rho u_x, \rho u_y, \rho e)$$

**Conversion formulas:**

From primitive to conservative:
$$\begin{align}
\rho_{cons} &= \rho \\
(\rho u_x)_{cons} &= \rho \cdot u_x \\
(\rho u_y)_{cons} &= \rho \cdot u_y \\
(\rho e)_{cons} &= \rho \left( \frac{u_x^2 + u_y^2}{2} + C_V T \right)
\end{align}$$

Where:
$$e = \frac{1}{2}(u_x^2 + u_y^2) + C_V T = \frac{1}{2}(u_x^2 + u_y^2) + \frac{R}{\gamma-1}T$$

**Why needed:** The conservation laws are solved in **conservative form**:
$$\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F} = \text{Source terms}$$

Not in primitive form:
$$\frac{\partial}{\partial t}\begin{pmatrix} \rho \\ u_x \\ u_y \\ p \end{pmatrix}$$

---

### Line 36: Apply Boundary Conditions

```python
prob.bc_conserved()
```

**Function:** `bc_conserved()` method

**Purpose:** Enforce boundary conditions on conservative variables

**Boundary condition type:** **Periodic** (for Taylor-Green vortex)

**What happens:**
```python
# Pseudocode
prob.rho[:, 0] = prob.rho[:, -1]    # Left = Right (periodic in y)
prob.rho[0, :] = prob.rho[-1, :]    # Bottom = Top (periodic in x)
# Same for all conservative variables
```

**Mathematical form:** Periodic BC
$$u(0, y) = u(L_x, y), \quad u(x, 0) = u(x, L_y)$$

**Differential equation context:** These are **boundary conditions** for the PDE:

$$\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = \mathbf{S}$$

$$\text{Subject to: } \mathbf{U}(x=0,y) = \mathbf{U}(x=2\pi,y) \text{ (periodic)}$$

---

## SECTION 5: DATA I/O INITIALIZATION

### Lines 42-48: Initialize Output Management

```python
# Initialize the data I/O
data_io = DataIO(prob, params)

if prob.grid.rank == 0:
    # Generate the output directory
    data_io.gen_path()
```

**Block Purpose:** Set up file saving system

**Line-by-line:**

| Line | Function | Purpose |
|------|----------|---------|
| `data_io = DataIO(prob, params)` | Constructor | Create I/O manager object |
| `if prob.grid.rank == 0:` | Parallel check | Only rank 0 (master process) creates directory |
| `data_io.gen_path()` | Path generation | Create `output/test/` directory if it doesn't exist |

**What gets created:**
```
output/
└── test/
    ├── fields_0000.h5      # Velocity, pressure, density
    ├── fields_0001.h5
    └── global_quantities.h5 # Kinetic energy, enstrophy, etc.
```

**No differential equations involved** - I/O operation

---

## SECTION 6: TIME EVOLUTION SETUP

### Lines 51-52: Initialize Time Integrator

```python
# Initialize the time evolution
evo = Time_Evolution(prob, data_io, params)
```

**Function:** `Time_Evolution(prob, data_io, params)` constructor

**Purpose:** Create the time-stepping engine

**What it does:**
1. Stores reference to problem object
2. Prepares for iterative time advancement
3. Sets up flux computation (reconstruction scheme)
4. Prepares viscous term calculation

**Differential equations prepared for:**

The solver will advance:
$$\frac{\partial \mathbf{U}}{\partial t} = -\frac{\partial \mathbf{F}_{inv}}{\partial x} - \frac{\partial \mathbf{F}_{inv}}{\partial y} + \frac{\partial \mathbf{F}_{visc,x}}{\partial x} + \frac{\partial \mathbf{F}_{visc,y}}{\partial y}$$

Where:
- $\mathbf{F}_{inv}$ = Inviscid fluxes (computed with TENO5 reconstruction)
- $\mathbf{F}_{visc}$ = Viscous fluxes (central differences)

---

## SECTION 7: TIME INTEGRATION LOOP

### Lines 55-62: Measure Time & Run Simulation

```python
# Time advance
t1 = time.time()
if para.scheme == 'RK2':
    evo.time_advance_rk2()
elif para.scheme == 'RK3':
    evo.time_advance_rk3()   
t2 = time.time()
```

**Block Purpose:** Execute time stepping and measure wall-clock time

**Line-by-line:**

| Line | Function | Purpose |
|------|----------|---------|
| `t1 = time.time()` | Get current time | Record start time (seconds since epoch) |
| `if para.scheme == 'RK2':` | Check scheme selection | Conditional based on para2d.py setting |
| `evo.time_advance_rk2()` | Execute RK2 stepping | 2nd-order Runge-Kutta time integration |
| `elif para.scheme == 'RK3':` | Alternative condition | Our case: `scheme = 'RK3'` |
| `evo.time_advance_rk3()` | Execute RK3 stepping | 3rd-order Runge-Kutta time integration |
| `t2 = time.time()` | Get final time | Record end time |

**What `evo.time_advance_rk3()` does (conceptually):**

It executes a loop like:
```python
for n in range(1, n_steps):
    # 1000 iterations with dt = 2.5e-3
    # Computes fluxes at cell faces
    # Updates conserved variables
    # Saves data at specified intervals
    # Advances from t^n to t^(n+1)
```

**Differential equations being solved:**

At each time step, solves the system of conservation laws:

$$\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u_x)}{\partial x} + \frac{\partial (\rho u_y)}{\partial y} = 0$$

$$\frac{\partial (\rho u_x)}{\partial t} + \frac{\partial (\rho u_x^2 + p)}{\partial x} + \frac{\partial (\rho u_x u_y)}{\partial y} = \frac{\partial \tau_{xx}}{\partial x} + \frac{\partial \tau_{xy}}{\partial y}$$

$$\frac{\partial (\rho u_y)}{\partial t} + \frac{\partial (\rho u_x u_y)}{\partial x} + \frac{\partial (\rho u_y^2 + p)}{\partial y} = \frac{\partial \tau_{xy}}{\partial x} + \frac{\partial \tau_{yy}}{\partial y}$$

$$\frac{\partial (\rho e)}{\partial t} + \frac{\partial [(\rho e + p) u_x]}{\partial x} + \frac{\partial [(\rho e + p) u_y]}{\partial y} = \text{Viscous work terms}$$

**RK3 time discretization (pseudo-code):**

```
U^(0) = U^n

U^(1) = U^(0) + dt * RHS(U^(0))

U^(2) = (3/4)*U^n + (1/4)*U^(1) + (1/4)*dt*RHS(U^(1))

U^(n+1) = (1/3)*U^n + (2/3)*U^(2) + (2/3)*dt*RHS(U^(2))
```

Where RHS is the spatial discretization using TENO5 fluxes and 2nd-order viscous terms.

---

### Lines 65-66: Print Elapsed Time

```python
if prob.grid.rank == 0:
    # Print the time taken
    print('\nTotal time of the simulation =', t2-t1, 'seconds\n')
```

**Block Purpose:** Display wall-clock runtime

**Line-by-line:**

| Line | Function | Purpose |
|------|----------|---------|
| `if prob.grid.rank == 0:` | Parallel check | Only master process (rank 0) prints |
| `print(...)` | Output function | Display total seconds elapsed |

**Example output:**
```
Total time of the simulation = 127.45 seconds
```

**What this tells us:**
- 1000 iterations on 256² grid took ~127 seconds
- Average: 0.127 seconds per iteration
- Useful for performance monitoring

**No differential equations involved** - output operation

---

---

# SUMMARY: DIFFERENTIAL EQUATIONS SOLVED

## Complete 2D Compressible Navier-Stokes System

### Conservative Form (What the Solver Uses)

$$\frac{\partial}{\partial t}\begin{pmatrix} \rho \\ \rho u_x \\ \rho u_y \\ \rho e \end{pmatrix} + \frac{\partial}{\partial x}\begin{pmatrix} \rho u_x \\ \rho u_x^2 + p \\ \rho u_x u_y \\ (\rho e + p)u_x \end{pmatrix} + \frac{\partial}{\partial y}\begin{pmatrix} \rho u_y \\ \rho u_x u_y \\ \rho u_y^2 + p \\ (\rho e + p)u_y \end{pmatrix} = \begin{pmatrix} 0 \\ \tau_{xx,x} + \tau_{xy,y} \\ \tau_{xy,x} + \tau_{yy,y} \\ q_x + q_y + \tau \cdot \mathbf{u} \end{pmatrix}$$

### Non-Conservative Form (Physical Interpretation)

**Continuity (Mass Conservation):**
$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$$

**Momentum (Newton's 2nd Law):**
$$\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) = -\nabla p + \nabla \cdot \boldsymbol{\tau}$$

Or in primitive form:
$$\rho \frac{D\mathbf{u}}{Dt} = -\nabla p + \nabla \cdot \boldsymbol{\tau}$$

**Energy (1st Law of Thermodynamics):**
$$\frac{\partial (\rho e)}{\partial t} + \nabla \cdot (\rho e \mathbf{u}) = -\nabla \cdot (p\mathbf{u}) + \nabla \cdot (\boldsymbol{\tau} \cdot \mathbf{u})$$

### Closure Relations

**Viscous Stress (Newtonian fluid):**
$$\boldsymbol{\tau} = \mu \left[\nabla \mathbf{u} + (\nabla \mathbf{u})^T - \frac{2}{3}(\nabla \cdot \mathbf{u})\mathbf{I}\right]$$

In 2D with constant viscosity:
$$\tau_{xx} = \mu \left(2\frac{\partial u_x}{\partial x} - \frac{2}{3}\nabla \cdot \mathbf{u}\right)$$
$$\tau_{yy} = \mu \left(2\frac{\partial u_y}{\partial y} - \frac{2}{3}\nabla \cdot \mathbf{u}\right)$$
$$\tau_{xy} = \mu \left(\frac{\partial u_x}{\partial y} + \frac{\partial u_y}{\partial x}\right)$$

**Equation of State (Ideal Gas):**
$$p = \rho R T = (\gamma - 1)\rho C_V T = (\gamma - 1)\rho e_{int}$$

Where:
$$e = \frac{1}{2}|\mathbf{u}|^2 + e_{int} = \frac{1}{2}(u_x^2 + u_y^2) + \frac{R}{\gamma - 1}T$$

---

## Initial Conditions (Taylor-Green Vortex at t=0)

$$u_x(x,y,0) = \sin(x)\cos(y)$$
$$u_y(x,y,0) = -\cos(x)\sin(y)$$
$$p(x,y,0) = R + \frac{1}{16}[\cos(2x) + \cos(2y)]$$
$$T(x,y,0) = \frac{p(x,y,0)}{R}$$

Density computed from $p = \rho R T$ with initial $T$

## Boundary Conditions

**Periodic (all four boundaries):**
$$u(0, y) = u(2\pi, y), \quad u(x, 0) = u(x, 2\pi)$$
$$p(0, y) = p(2\pi, y), \quad p(x, 0) = p(x, 2\pi)$$

## Numerical Discretization

- **Spatial:** TENO5 reconstruction + 2nd-order central differences for viscous terms
- **Temporal:** RK3 (3rd-order Runge-Kutta)
- **Grid:** Uniform, $\Delta x = \Delta y = \frac{2\pi}{256}$
- **CFL:** $C = 0.5 < C_{max} \approx 1.256$

## Key Physical Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $\gamma$ | 1.4 | Ratio of specific heats (air) |
| $Re$ | 1600 | Reynolds number |
| $Ma$ | 1.25 | Mach number |
| $\mu$ | Constant | Dynamic viscosity |

---

## Expected Physics Behavior

1. **Viscous Decay:** Taylor-Green vortex decays exponentially at rate $e^{-\nu k^2 t}$
2. **Kinetic Energy:** $KE(t) \propto e^{-2Re \cdot t}$ (theoretical for low Ma)
3. **Enstrophy:** $\Omega(t) = \int (\nabla \times \mathbf{u})^2 dA$ also decays
4. **Periodic Boundaries:** No boundary layer effects, pure bulk dissipation

