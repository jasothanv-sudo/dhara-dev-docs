# DHARA Differential Equations Reference

A focused guide to all mathematical equations solved by the 2D solver.

---

## 1. GOVERNING EQUATIONS

### 1.1 Conservative Form (What the Code Solves)

This is the form used internally by DHARA:

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}_x}{\partial x} + \frac{\partial \mathbf{F}_y}{\partial y} = \mathbf{S}$$

Where:

**State vector (conservative variables):**
$$\mathbf{U} = \begin{pmatrix} \rho \\ \rho u_x \\ \rho u_y \\ \rho e \end{pmatrix}$$

**Inviscid fluxes in x-direction:**
$$\mathbf{F}_x = \begin{pmatrix} \rho u_x \\ \rho u_x^2 + p \\ \rho u_x u_y \\ (\rho e + p)u_x \end{pmatrix}$$

**Inviscid fluxes in y-direction:**
$$\mathbf{F}_y = \begin{pmatrix} \rho u_y \\ \rho u_x u_y \\ \rho u_y^2 + p \\ (\rho e + p)u_y \end{pmatrix}$$

**Source (viscous) terms:**
$$\mathbf{S} = \begin{pmatrix} 0 \\ \frac{\partial \tau_{xx}}{\partial x} + \frac{\partial \tau_{xy}}{\partial y} \\ \frac{\partial \tau_{xy}}{\partial x} + \frac{\partial \tau_{yy}}{\partial y} \\ \frac{\partial}{\partial x}(\tau_{xx} u_x + \tau_{xy} u_y) + \frac{\partial}{\partial y}(\tau_{xy} u_x + \tau_{yy} u_y) \end{pmatrix}$$

---

### 1.2 Primitive Form (Physical Interpretation)

This is easier to understand:

**Continuity Equation (Mass Conservation):**
$$\frac{\partial \rho}{\partial t} + \frac{\partial(\rho u_x)}{\partial x} + \frac{\partial(\rho u_y)}{\partial y} = 0$$

**X-Momentum Equation:**
$$\frac{\partial(\rho u_x)}{\partial t} + \frac{\partial(\rho u_x^2)}{\partial x} + \frac{\partial(\rho u_x u_y)}{\partial y} = -\frac{\partial p}{\partial x} + \frac{\partial \tau_{xx}}{\partial x} + \frac{\partial \tau_{xy}}{\partial y}$$

Or in non-conservative form:
$$\rho \frac{\partial u_x}{\partial t} + \rho u_x \frac{\partial u_x}{\partial x} + \rho u_y \frac{\partial u_x}{\partial y} = -\frac{\partial p}{\partial x} + \frac{\partial \tau_{xx}}{\partial x} + \frac{\partial \tau_{xy}}{\partial y}$$

**Y-Momentum Equation:**
$$\frac{\partial(\rho u_y)}{\partial t} + \frac{\partial(\rho u_x u_y)}{\partial x} + \frac{\partial(\rho u_y^2)}{\partial y} = -\frac{\partial p}{\partial y} + \frac{\partial \tau_{xy}}{\partial x} + \frac{\partial \tau_{yy}}{\partial y}$$

**Energy Equation:**
$$\frac{\partial(\rho e)}{\partial t} + \frac{\partial[(\rho e + p)u_x]}{\partial x} + \frac{\partial[(\rho e + p)u_y]}{\partial y} = \frac{\partial}{\partial x}(\tau_{xx} u_x + \tau_{xy} u_y) + \frac{\partial}{\partial y}(\tau_{xy} u_x + \tau_{yy} u_y)$$

---

## 2. CLOSURE RELATIONS

### 2.1 Viscous Stress Tensor (Newtonian Fluid)

For a Newtonian incompressible fluid:

$$\boldsymbol{\tau} = \mu \left[\nabla \mathbf{u} + (\nabla \mathbf{u})^T\right] = \mu \begin{pmatrix} 2\frac{\partial u_x}{\partial x} & \frac{\partial u_x}{\partial y} + \frac{\partial u_y}{\partial x} \\ \frac{\partial u_y}{\partial x} + \frac{\partial u_x}{\partial y} & 2\frac{\partial u_y}{\partial y} \end{pmatrix}$$

With bulk viscosity correction (for compressible flow):

$$\tau_{xx} = \mu\left[2\frac{\partial u_x}{\partial x} - \frac{2}{3}(\nabla \cdot \mathbf{u})\right]$$

$$\tau_{yy} = \mu\left[2\frac{\partial u_y}{\partial y} - \frac{2}{3}(\nabla \cdot \mathbf{u})\right]$$

$$\tau_{xy} = \tau_{yx} = \mu\left[\frac{\partial u_x}{\partial y} + \frac{\partial u_y}{\partial x}\right]$$

Where:
$$\nabla \cdot \mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y}$$

**In para2d.py:** `mu_varying = False` → constant dynamic viscosity $\mu$

**Kinematic viscosity:**
$$\nu = \frac{\mu}{\rho}$$

---

### 2.2 Equation of State (Ideal Gas)

$$p = \rho R T$$

Where:
- $R = \frac{R_u}{M}$ (specific gas constant)
- For air: $R \approx 287$ J/(kg·K)

Also written in terms of energy:
$$p = (\gamma - 1)\rho e_{int} = (\gamma - 1)\rho C_V T$$

Where:
$$C_V = \frac{R}{\gamma - 1}$$

**Total energy per unit mass:**
$$e = \frac{1}{2}(u_x^2 + u_y^2) + e_{int} = \frac{1}{2}(u_x^2 + u_y^2) + C_V T$$

---

### 2.3 Reynolds Number

$$Re = \frac{\rho u_0 L}{\mu}$$

Where:
- $u_0$ = reference velocity (Taylor-Green: $u_0 \sim 1$)
- $L$ = reference length (domain size: $L = 2\pi$)
- $\rho$ = reference density
- $\mu$ = dynamic viscosity

**Inverse relationship with viscosity:**
$$\mu = \frac{\rho u_0 L}{Re}$$

For `Re = 1600` in para2d.py:
$$\mu = \frac{\rho \cdot 1 \cdot 2\pi}{1600} \approx 3.93 \times 10^{-3} \rho$$

---

### 2.4 Mach Number

$$Ma = \frac{u_0}{a}$$

Where speed of sound:
$$a = \sqrt{\gamma R T}$$

For ideal gas at reference temperature $T_0$:
$$a_0 = \sqrt{\gamma R T_0}$$

**In para2d.py:** `Ma = 1.25` (transonic flow)

For Taylor-Green at $T_0 = 1$:
$$a_0 = \sqrt{\gamma R} = \sqrt{1.4 \times R}$$

So reference velocity:
$$u_0 = Ma \cdot a_0 = 1.25 \sqrt{\gamma R}$$

---

## 3. INITIAL CONDITIONS

### 3.1 Taylor-Green Vortex (para2d.py, line 27-33)

**Velocity:**
$$u_x(x,y,0) = \sin(x) \cos(y)$$
$$u_y(x,y,0) = -\cos(x) \sin(y)$$
$$u_z(x,y,0) = 0 \quad \text{(2D)}$$

**Verification - Continuity at t=0:**
$$\nabla \cdot \mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} = \cos(x)\cos(y) - \cos(x)\cos(y) = 0 \quad \checkmark$$

Flow is **divergence-free** (approximately incompressible).

**Pressure:**
$$p(x,y,0) = p_{ref} + \frac{1}{16}[\cos(2x) + \cos(2y)]$$

Where $p_{ref} = R$ (gas constant / dimensional reference).

**Temperature:**
$$T(x,y,0) = \frac{p(x,y,0)}{\rho(x,y,0) \cdot R}$$

**Density:**
From $p = \rho R T$:
$$\rho(x,y,0) = \frac{p(x,y,0)}{R T(x,y,0)}$$

At $t=0$: All fields are smooth, periodic, and analytical.

---

### 3.2 Vorticity Field

For Taylor-Green, vorticity (2D, scalar):
$$\omega(x,y,0) = \frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}$$
$$= \sin(x)\sin(y) + \sin(x)\sin(y) = 2\sin(x)\sin(y)$$

**Enstrophy (integrated vorticity²):**
$$\Omega(0) = \int_0^{2\pi}\int_0^{2\pi} \omega^2 \, dx\, dy = 4\pi^2$$

---

### 3.3 Kinetic Energy

**Definition:**
$$KE(t) = \frac{1}{2}\int_D \rho(u_x^2 + u_y^2) \, dA$$

**Initial value (from analytical IC):**
$$KE(0) = \frac{1}{2}\int_0^{2\pi}\int_0^{2\pi} [\sin^2(x)\cos^2(y) + \cos^2(x)\sin^2(y)] \, dx\, dy$$

By symmetry and orthogonality:
$$KE(0) = \pi^2 \approx 9.87$$

---

## 4. BOUNDARY CONDITIONS

### 4.1 Periodic Boundary Conditions

Applied to all dependent variables:

**X-direction (periodic):**
$$\mathbf{U}(0, y, t) = \mathbf{U}(L_x, y, t)$$

Where $L_x = 2\pi$

**Y-direction (periodic):**
$$\mathbf{U}(x, 0, t) = \mathbf{U}(x, L_y, t)$$

Where $L_y = 2\pi$

**Implemented by:** `prob.bc_conserved()` in main2d.py (line 36)

**Effect:** No boundary layers, pure bulk dissipation

---

## 5. VISCOUS DECAY THEORY

### 5.1 Analytical Decay for Taylor-Green

For incompressible Taylor-Green vortex:

$$u_x(x,y,t) = e^{-2\nu t} \sin(x)\cos(y)$$
$$u_y(x,y,t) = -e^{-2\nu t} \cos(x)\sin(y)$$

**Decay exponent:** $-2\nu$ where $\nu = \mu/\rho$

---

### 5.2 Kinetic Energy Decay

**Analytical (incompressible limit):**
$$KE(t) = KE(0) \cdot e^{-2\nu k^2 t}$$

Where $k = 1$ (dominant wavenumber for Taylor-Green).

$$KE(t) = KE(0) \cdot e^{-2\nu t}$$

**With Reynolds number:**
$$KE(t) = KE(0) \cdot \exp\left(-\frac{2}{\text{Re}} t \right)$$

Actually, for Taylor-Green in compressible form, decay is more complex due to sound waves and interactions between modes.

---

### 5.3 Expected Decay Rate (para2d.py)

**Parameters:**
- $Re = 1600$
- $\nu = L_0 u_0 / Re = 2\pi \cdot 1 / 1600 \approx 3.93 \times 10^{-3}$
- Dominant wavenumber: $k = 1$

**Decay rate:**
$$\lambda = 2\nu k^2 = 2 \times 3.93 \times 10^{-3} \times 1 \approx 7.85 \times 10^{-3} \text{ s}^{-1}$$

**At final time** $t = 2.5$:
$$\frac{KE(2.5)}{KE(0)} \approx e^{-7.85 \times 10^{-3} \times 2.5} = e^{-0.0196} \approx 0.981$$

From dominant mode alone: **2% dissipation**

However, higher modes decay faster:
$$KE(t) = \sum_k KE_k(0) \cdot e^{-2\nu k^2 t}$$

For $k=2$: rate = $2\nu \times 4 = 4 \times 7.85 \times 10^{-3} = 0.0314$

At $t=2.5$: $e^{-0.0785} \approx 0.925$ (7.5% loss from k=2 mode)

**Overall** kinetic energy decays non-exponentially due to multi-mode structure.

---

## 6. NUMERICAL DISCRETIZATION EQUATIONS

### 6.1 Spatial Discretization - TENO5

**General principle:** Approximate $\frac{\partial f}{\partial x}$ at cell face $i+1/2$

$$\left.\frac{\partial f}{\partial x}\right|_{i+1/2} \approx \frac{f_{i+1/2}^{recon} - f_{i-1/2}^{recon}}{\Delta x}$$

**TENO5 reconstruction:** Uses 5-point stencil
$$f_{i+1/2}^{recon} = w_0 f_{i-2:i+2}^{(0)} + w_1 f_{i-2:i+2}^{(1)} + w_2 f_{i-2:i+2}^{(2)}$$

Where $(0), (1), (2)$ are three candidate stencils (each 5-point).

Weights $w_0, w_1, w_2$ are adapted based on smoothness (TENO weighting).

**Order of accuracy:** 5th-order in smooth regions

---

### 6.2 Spatial Discretization - Viscous Terms

**2nd-order central difference:**
$$\frac{\partial^2 u}{\partial x^2}\bigg|_i \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$$

**Error:** $O(\Delta x^2)$

**Implemented as:** `viscous_term_order = 2` in para2d.py

---

### 6.3 Temporal Discretization - RK3

**General ODE:** $\frac{d\mathbf{U}}{dt} = \mathbf{F}(t, \mathbf{U})$

**RK3 stages:**

**Stage 1 (explicit):**
$$\mathbf{U}^{(1)} = \mathbf{U}^n + \Delta t \, \mathbf{F}(\mathbf{U}^n)$$

**Stage 2 (combination):**
$$\mathbf{U}^{(2)} = \frac{3}{4}\mathbf{U}^n + \frac{1}{4}\mathbf{U}^{(1)} + \frac{1}{4}\Delta t \, \mathbf{F}(\mathbf{U}^{(1)})$$

**Stage 3 (final):**
$$\mathbf{U}^{n+1} = \frac{1}{3}\mathbf{U}^n + \frac{2}{3}\mathbf{U}^{(2)} + \frac{2}{3}\Delta t \, \mathbf{F}(\mathbf{U}^{(2)})$$

**Order of accuracy:** 3rd-order $O(\Delta t^3)$

**Stability:** CFL condition
$$C = \max(|u|) \frac{\Delta t}{\Delta x} \leq C_{max}$$

For RK3: $C_{max} \approx 1.256$

**In para2d.py:** `cfl_cons = 0.5` → Conservative (well below limit)

$$C = 1.25 \times \frac{2.5 \times 10^{-3}}{2\pi/256} \approx 0.128 \ll 1.256 \quad \checkmark$$

---

## 7. ERROR ANALYSIS

### 7.1 Local Truncation Error (LTE)

**Spatial:** 
$$LTE_x = O(\Delta x^5) \quad \text{(TENO5)}$$
$$LTE_y = O(\Delta y^5) \quad \text{(TENO5)}$$

**Temporal:**
$$LTE_t = O(\Delta t^3) \quad \text{(RK3)}$$

**Total LTE:**
$$LTE = O(\Delta t^3) + O(\Delta x^5 + \Delta y^5)$$

Since $\Delta x = \Delta y = 2\pi/256 \approx 0.0245$ (smaller than $\Delta t = 2.5 \times 10^{-3}$):
$$LTE \approx O(\Delta t^3)$$

**Overall accuracy:** Limited by temporal scheme → **3rd-order**

---

### 7.2 Grid Refinement Study

**Grid points:** $N = 256$
$$\Delta x = \frac{2\pi}{256} \approx 0.02454$$

**Error at final time (rough estimate for smooth problem):**
$$\text{Error} \sim O(\Delta t^3) + O(\Delta x^5) = O((2.5e-3)^3) + O((0.0245)^5)$$
$$\approx 1.56 \times 10^{-8} + 8.3 \times 10^{-10}$$

Dominated by temporal error.

To improve: reduce $\Delta t$ or increase RK order (→ RK5)

---

## 8. SUMMARY TABLE

| Equation | Form | Domain | BC |
|----------|------|--------|-----|
| Continuity | $\partial \rho/\partial t + \nabla \cdot (\rho \mathbf{u})= 0$ | $[0,2\pi]^2$ | Periodic |
| X-Momentum | $\partial(\rho u_x)/\partial t + \nabla \cdot (\rho u_x \mathbf{u}) = -\partial p/\partial x + \nabla \cdot \boldsymbol{\tau}$ | $[0,2\pi]^2$ | Periodic |
| Y-Momentum | $\partial(\rho u_y)/\partial t + \nabla \cdot (\rho u_y \mathbf{u}) = -\partial p/\partial y + \nabla \cdot \boldsymbol{\tau}$ | $[0,2\pi]^2$ | Periodic |
| Energy | $\partial(\rho e)/\partial t + \nabla \cdot (\rho e \mathbf{u}) = -\nabla \cdot (p\mathbf{u}) + \nabla \cdot (\boldsymbol{\tau} \cdot \mathbf{u})$ | $[0,2\pi]^2$ | Periodic |
| EOS | $p = \rho R T = (\gamma-1)\rho e_{int}$ | Everywhere | — |
| Stress | $\tau_{ij} = \mu[\partial u_i/\partial x_j + \partial u_j/\partial x_i - (2/3)\delta_{ij}\nabla \cdot \mathbf{u}]$ | Everywhere | — |

| Parameter | Symbol | Value | Meaning |
|-----------|--------|-------|---------|
| Grid points | $N_x, N_y$ | 256, 256 | Resolution |
| Time step | $\Delta t$ | 0.0025 | Temporal interval |
| Domain | $[L_{min}, L_{max}]$ | $[-\pi, \pi]^2$ | Computational box |
| Grid spacing | $\Delta x, \Delta y$ | 0.02454 | Spatial interval |
| Specific heat ratio | $\gamma$ | 1.4 | Thermodynamic property |
| Reynolds number | $Re$ | 1600 | Viscous parameter |
| Mach number | $Ma$ | 1.25 | Compressibility parameter |
| Final time | $t_f$ | 2.5 | Simulation duration |
| Total steps | $n_{steps}$ | 1000 | Iterations |

