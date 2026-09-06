# Computational Mathematics for ML: Calculus + Linear Algebra Course Outline

## Goal

Learn calculus deeply enough to reason about machine learning,
optimization, dynamical systems, and mathematical models **without
making hand calculation the main activity**.

The computer handles most algebra. Your job is to:

> **Predict → Visualize → Perturb → Compute → Interpret → Experiment**

### Core Python stack

-   **PyTorch** --- automatic differentiation, gradients, Jacobians,
    Hessians, optimization
-   **SymPy** --- symbolic mathematics and exact derivatives/integrals
-   **NumPy** --- numerical experiments and arrays
-   **SciPy** --- numerical differentiation, integration, optimization,
    and differential equations
-   **Matplotlib** --- 2D/3D visualization and geometric intuition

### Scope

Approximately **600 progressively harder problems** across:

-   Calculus I
-   Calculus II
-   Multivariable Calculus / Calculus III
-   Vector Calculus
-   Optimization
-   Differential Equations
-   Calculus for Machine Learning

------------------------------------------------------------------------

# Level 0 --- Mathematical Foundations

## Module 0.1 --- Numbers, Variables, and Expressions

Topics:

-   Real numbers
-   Variables and constants
-   Expressions
-   Equations
-   Inequalities
-   Absolute value
-   Intervals
-   Scientific notation

The objective is not extensive algebra practice. Learn enough notation
and manipulation to understand the calculus that follows.

## Module 0.2 --- Functions

The most important prerequisite.

\[ y=f(x) \]

Topics:

-   Inputs and outputs
-   Domain and range
-   Function composition
-   Inverse functions
-   Piecewise functions
-   Polynomial functions
-   Rational functions
-   Exponential functions
-   Logarithms
-   Trigonometric functions
-   Sigmoid-like functions

Core intuition:

> A function is a transformation from one space into another.

**Target: \~30 exercises**

------------------------------------------------------------------------

# Level 1 --- Single-Variable Calculus

## Module 1 --- Limits and Continuity

\[ `\lim`{=tex}\_{x`\to`{=tex}a} f(x) \]

Topics:

-   Approaching a value
-   Left and right limits
-   Infinite limits
-   Limits at infinity
-   Continuity
-   Discontinuities
-   Asymptotic behavior
-   Numerical instability

Computational approach:

-   Evaluate functions increasingly close to a point
-   Plot behavior near discontinuities
-   Compare numerical observations with SymPy limits

**Libraries:** NumPy, SymPy, Matplotlib

**Target: \~25 exercises**

------------------------------------------------------------------------

## Module 2 --- Derivatives

Central question:

> How sensitive is the output to a tiny change in the input?

Start experimentally with:

\[ `\frac{f(x+h)-f(x)}{h}`{=tex} \]

Then shrink (h) and develop the limit intuition:

\[ f'(x)=`\lim`{=tex}\_{h`\to0`{=tex}}`\frac{f(x+h)-f(x)}{h}`{=tex} \]

Topics:

-   Slope
-   Secant lines
-   Tangent lines
-   Instantaneous rate of change
-   Derivative as a function
-   Numerical differentiation
-   Symbolic differentiation
-   Automatic differentiation
-   Higher derivatives

Compare three approaches:

-   **SciPy** → numerical differentiation
-   **SymPy** → symbolic differentiation
-   **PyTorch** → automatic differentiation

**Target: \~40 exercises**

------------------------------------------------------------------------

## Module 3 --- Differentials

Learn:

\[ dy=f'(x),dx \]

as a local prediction.

For a small change (dx):

\[ `\Delta`{=tex}y `\approx`{=tex}dy=f'(x)dx \]

Topics:

-   Small-change intuition
-   Linear approximation
-   Sensitivity
-   Propagation of small changes
-   Approximation error
-   Local versus global behavior

Core intuition:

> The derivative tells you how a tiny perturbation of the input
> propagates to the output.

**Target: \~25 exercises**

------------------------------------------------------------------------

## Module 4 --- Chain Rule

One of the most important modules for machine learning.

For a computational chain

\[ x`\rightarrow`{=tex}u`\rightarrow`{=tex}v`\rightarrow`{=tex}y \]

understand:

\[ `\frac{dy}{dx}`{=tex} = `\frac{dy}{dv}`{=tex} `\frac{dv}{du}`{=tex}
`\frac{du}{dx}`{=tex} \]

Topics:

-   Function composition
-   Local derivatives
-   Computational graphs
-   Forward sensitivity
-   Backward sensitivity
-   Automatic differentiation
-   Foundations of backpropagation

Use PyTorch to build computational graphs and trace gradients backward.

**Target: \~40 exercises**

------------------------------------------------------------------------

## Module 5 --- Taylor Series and Local Approximation

First-order approximation:

\[ f(x+`\Delta`{=tex}x) `\approx`{=tex} f(x)+f'(x)`\Delta`{=tex}x \]

Second-order approximation:

\[ f(x+`\Delta`{=tex}x) `\approx`{=tex} f(x)+f'(x)`\Delta`{=tex}x
+`\frac{1}{2}`{=tex}f''(x)(`\Delta`{=tex}x)\^2 \]

General Taylor series:

\[ f(x) = `\sum`{=tex}\_{n=0}\^{`\infty`{=tex}}
`\frac{f^{(n)}(a)}{n!}`{=tex}(x-a)\^n \]

Topics:

-   Linear approximations
-   Quadratic approximations
-   Polynomial approximation
-   Approximation error
-   Local models of nonlinear functions
-   Relationship to optimization
-   Newton's method intuition

**Target: \~30 exercises**

------------------------------------------------------------------------

# Level 2 --- Integral Calculus

## Module 6 --- Integration

Switch perspective:

> A derivative asks what happens locally. An integral asks what happens
> when tiny effects accumulate.

Start with sums:

\[ `\sum`{=tex}\_i f(x_i)`\Delta`{=tex}x \]

Then develop:

\[ `\int`{=tex}\_a\^b f(x),dx \]

Topics:

-   Area
-   Signed area
-   Accumulation
-   Riemann sums
-   Definite integrals
-   Numerical integration
-   Indefinite integrals
-   Fundamental Theorem of Calculus

**Primary library:** `scipy.integrate`

Use SymPy for symbolic solutions when useful.

**Target: \~35 exercises**

------------------------------------------------------------------------

## Module 7 --- Applications of Integration

Topics:

-   Accumulated change
-   Average value
-   Distance from velocity
-   Probability density
-   Expected values
-   Center of mass
-   Work
-   Volumes
-   Arc length

**Target: \~25 exercises**

------------------------------------------------------------------------

# Level 3 --- Multivariable Calculus

## Module 8 --- Functions of Multiple Variables

Move from

\[ f(x) \]

to

\[ f(x_1,x_2,`\ldots`{=tex},x_n) \]

Start visually with:

\[ z=f(x,y) \]

Topics:

-   Domains in higher dimensions
-   Surfaces
-   Level curves
-   Contour plots
-   Scalar fields
-   High-dimensional functions

Use 3D plots and contour maps extensively.

**Target: \~25 exercises**

------------------------------------------------------------------------

## Module 9 --- Partial Derivatives

Central question:

> What happens if I change one variable while holding everything else
> fixed?

\[ `\frac{\partial f}{\partial x}`{=tex} `\qquad`{=tex}
`\frac{\partial f}{\partial y}`{=tex} \]

Topics:

-   Partial derivatives
-   Holding variables constant
-   Sensitivity with multiple inputs
-   Higher-order partial derivatives
-   Mixed partial derivatives

Use PyTorch autograd to compute and experimentally verify partial
derivatives.

**Target: \~35 exercises**

------------------------------------------------------------------------

## Module 10 --- Gradients

Combine partial derivatives:

\[ `\nabla`{=tex}f=

```{=tex}
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
```
\]

Core geometric intuition:

> The gradient points in the direction of steepest increase.

Topics:

-   Gradient vectors
-   Gradient magnitude
-   Contour plots
-   Steepest ascent
-   Steepest descent
-   Sensitivity
-   Gradient fields

**Target: \~35 exercises**

------------------------------------------------------------------------

## Module 11 --- Directional Derivatives

For a direction (u):

\[ D_u f=`\nabla`{=tex}f`\cdot`{=tex}u \]

Interpretation:

> What happens to the output if I move in this particular direction?

Topics:

-   Unit direction vectors
-   Directional change
-   Gradient projection
-   Maximum directional derivative
-   Relationship between gradients and geometry

Experiment with random directions numerically.

**Target: \~20 exercises**

------------------------------------------------------------------------

## Module 12 --- Total Differentials

Generalize the differential:

\[ df= `\frac{\partial f}{\partial x}`{=tex}dx+
`\frac{\partial f}{\partial y}`{=tex}dy \]

More generally:

\[ df=`\nabla`{=tex}f\^`\top`{=tex}dx \]

Core intuition:

> Gradient × tiny input change ≈ output change.

Topics:

-   Total differentials
-   Multivariable linearization
-   Error propagation
-   Sensitivity analysis
-   Local approximation

**Target: \~25 exercises**

------------------------------------------------------------------------

## Module 13 --- Jacobians

Move from scalar-valued functions to vector-valued functions:

\[ f:`\mathbb{R}`{=tex}^n`\rightarrow`{=tex}`\mathbb{R}`{=tex}^m \]

The derivative becomes a matrix:

\[ J=

```{=tex}
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1}&\cdots&
\frac{\partial f_1}{\partial x_n}\\
\vdots&&\vdots\\
\frac{\partial f_m}{\partial x_1}&\cdots&
\frac{\partial f_m}{\partial x_n}
\end{bmatrix}
```
\]

Topics:

-   Jacobian matrices
-   Jacobian as a local linear transformation
-   Shapes and dimensions
-   Jacobian-vector products
-   Vector-Jacobian products
-   Forward-mode autodiff
-   Reverse-mode autodiff
-   Coordinate transformations
-   Relationship to backpropagation

Useful PyTorch tools:

-   `torch.func.jacrev`
-   `torch.func.jacfwd`
-   JVPs
-   VJPs

**Target: \~40 exercises**

------------------------------------------------------------------------

## Module 14 --- Hessians

Differentiate the gradient:

\[ H_f=

```{=tex}
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2}&\cdots\\
\vdots&\ddots
\end{bmatrix}
```
\]

Core intuition:

-   **Gradient** → slope information
-   **Hessian** → curvature information

Topics:

-   Second derivatives
-   Curvature
-   Mixed partial derivatives
-   Positive and negative curvature
-   Hessian eigenvalues
-   Local minima
-   Local maxima
-   Saddle points
-   Second-order Taylor approximations

Useful PyTorch tool:

-   `torch.func.hessian`

**Target: \~30 exercises**

------------------------------------------------------------------------

# Level 4 --- Optimization

## Module 15 --- Unconstrained Optimization

Start with gradient descent:

\[ x\_{t+1}=x_t-`\eta`{=tex}`\nabla`{=tex}f(x_t) \]

Topics:

-   Objective functions
-   Gradient descent
-   Learning rates
-   Convergence
-   Local minima
-   Global minima
-   Saddle points
-   Momentum
-   Newton's method
-   Hessian-based optimization
-   Convex functions
-   Nonconvex functions

Implement optimizers manually before comparing them with `torch.optim`.

**Target: \~40 exercises**

------------------------------------------------------------------------

## Module 16 --- Constrained Optimization

Solve:

\[ `\min`{=tex}f(x,y) \]

subject to:

\[ g(x,y)=c \]

Topics:

-   Constraints
-   Feasible regions
-   Lagrange multipliers
-   Geometric interpretation
-   Equality constraints
-   Inequality constraints
-   KKT intuition

Use SciPy optimization for computational experiments.

**Target: \~25 exercises**

------------------------------------------------------------------------

# Level 5 --- Vector Calculus

## Module 17 --- Vector Fields

Instead of a scalar output, study:

\[ F(x,y) =

```{=tex}
\begin{bmatrix}
u(x,y)\\
v(x,y)
\end{bmatrix}
```
\]

Topics:

-   Vector fields
-   Velocity fields
-   Force fields
-   Gradient fields
-   Flow fields
-   Trajectories through fields
-   Streamlines

Use quiver plots and stream plots heavily.

**Target: \~25 exercises**

------------------------------------------------------------------------

## Module 18 --- Multiple Integrals

Move from:

\[ `\int`{=tex}f(x),dx \]

to:

\[ `\iint`{=tex}f(x,y),dx,dy \]

and:

\[ `\iiint`{=tex}f(x,y,z),dV \]

Topics:

-   Double integrals
-   Triple integrals
-   Integration regions
-   Volume
-   Mass
-   Multivariable probability distributions
-   Expectations in multiple dimensions

**Primary library:** `scipy.integrate`

**Target: \~30 exercises**

------------------------------------------------------------------------

## Module 19 --- Coordinate Transformations

Topics:

-   Cartesian coordinates
-   Polar coordinates
-   Cylindrical coordinates
-   Spherical coordinates
-   Coordinate transformations
-   Jacobian determinant
-   Change of variables

Understand geometrically why an area element such as

\[ dx,dy \]

becomes:

\[ r,dr,d`\theta`{=tex} \]

under polar coordinates.

**Target: \~25 exercises**

------------------------------------------------------------------------

## Module 20 --- Line Integrals

Integrate along a path:

\[ `\int`{=tex}\_C F`\cdot`{=tex}dr \]

Interpretation:

> Accumulate what a vector field does while moving along a trajectory.

Topics:

-   Parametric curves
-   Line integrals
-   Work
-   Circulation
-   Path dependence
-   Path independence

Visualize the path and vector field for most exercises.

**Target: \~25 exercises**

------------------------------------------------------------------------

## Module 21 --- Surface Integrals

Move from paths to surfaces:

\[ `\iint`{=tex}\_S F`\cdot`{=tex}n,dS \]

Topics:

-   Parametric surfaces
-   Surface normals
-   Orientation
-   Flux
-   Surface area
-   Surface integration

**Target: \~25 exercises**

------------------------------------------------------------------------

## Module 22 --- Divergence

\[ `\nabla`{=tex}`\cdot`{=tex}F \]

Core intuition:

> Is the vector field locally flowing outward or inward?

Topics:

-   Sources
-   Sinks
-   Local expansion
-   Local compression
-   Divergence fields
-   Flux intuition

Visualize sources and sinks.

**Target: \~20 exercises**

------------------------------------------------------------------------

## Module 23 --- Curl

\[ `\nabla`{=tex}`\times`{=tex}F \]

Core intuition:

> If I placed a tiny paddle wheel here, would it rotate?

Topics:

-   Local rotation
-   Circulation
-   Curl fields
-   Curl-free fields
-   Conservative fields

**Target: \~20 exercises**

------------------------------------------------------------------------

# Level 6 --- The Big Vector Calculus Theorems

## Module 24 --- Conservative Fields

Understand:

\[ F=`\nabla`{=tex}`\phi`{=tex} \]

Topics:

-   Potential functions
-   Path independence
-   Conservative forces
-   Gradient fields
-   Curl and conservative fields

**Target: \~20 exercises**

------------------------------------------------------------------------

## Module 25 --- Green's Theorem

Connect behavior around a boundary with behavior throughout its
interior.

Conceptually:

\[ `\text{boundary information}`{=tex} `\longleftrightarrow`{=tex}
`\text{interior information}`{=tex} \]

Computational approach:

1.  Calculate the boundary integral numerically.
2.  Calculate the interior integral numerically.
3.  Compare the results.
4.  Visualize the vector field and region.
5.  Develop intuition for why they agree.

**Target: \~15 exercises**

------------------------------------------------------------------------

## Module 26 --- Stokes' Theorem

\[ `\oint`{=tex}\_{`\partial`{=tex}S}F`\cdot`{=tex}dr =
`\iint`{=tex}\_S(`\nabla`{=tex}`\times`{=tex}F)`\cdot`{=tex}n,dS \]

Approach:

-   Compute both sides numerically
-   Visualize the surface and boundary
-   Understand the relationship between local curl and boundary
    circulation
-   Only then study the formal mathematical statement in detail

**Target: \~15 exercises**

------------------------------------------------------------------------

## Module 27 --- Divergence Theorem

\[ `\iiint`{=tex}*V`\nabla`{=tex}`\cdot`{=tex}F,dV =
`\iint`{=tex}*{`\partial`{=tex}V}F`\cdot`{=tex}n,dS \]

Core relationship:

\[ `\text{divergence inside volume}`{=tex} `\longleftrightarrow`{=tex}
`\text{flux through boundary}`{=tex} \]

Approach:

-   Compute the volume integral
-   Compute the surface flux
-   Compare numerically
-   Visualize sources, sinks, and boundary flow

**Target: \~15 exercises**

------------------------------------------------------------------------

# Level 7 --- Differential Equations

## Module 28 --- Ordinary Differential Equations

Start with:

\[ `\frac{dy}{dt}`{=tex}=f(t,y) \]

Core intuition:

> If I know how the system changes locally, can I predict how it evolves
> globally?

Topics:

-   First-order ODEs
-   Initial value problems
-   Exponential growth and decay
-   Logistic equations
-   Second-order ODEs
-   Oscillators
-   Coupled ODEs
-   Phase space
-   Stability
-   Euler's method
-   Numerical ODE solvers

Primary tool:

-   `scipy.integrate.solve_ivp`

**Target: \~40 exercises**

------------------------------------------------------------------------

## Module 29 --- Dynamical Systems

Study systems such as:

\[ `\frac{dx}{dt}`{=tex}=f(x,y) \]

\[ `\frac{dy}{dt}`{=tex}=g(x,y) \]

Topics:

-   Phase portraits
-   Equilibrium points
-   Fixed points
-   Stability
-   Jacobians
-   Eigenvalues
-   Attractors
-   Nonlinear dynamics
-   Bifurcations
-   Chaos

This module connects Jacobians and eigenvalues to the behavior of
evolving systems.

**Target: \~30 exercises**

------------------------------------------------------------------------

## Module 30 --- Partial Differential Equations

Advanced endpoint.

### Heat equation

\[ `\frac{\partial u}{\partial t}`{=tex} =
`\alpha`{=tex}`\nabla`{=tex}\^2u \]

### Wave equation

\[ `\frac{\partial^2u}{\partial t^2}`{=tex} = c^2`\nabla`{=tex}^2u \]

### Laplace equation

\[ `\nabla`{=tex}\^2u=0 \]

Topics:

-   PDE intuition
-   Boundary conditions
-   Initial conditions
-   Discretization
-   Finite differences
-   Diffusion
-   Waves
-   Laplacian
-   Numerical PDE solutions

The objective is not a complete graduate PDE course. It is to understand
what PDEs describe and how computers solve them.

**Target: \~30 exercises**

------------------------------------------------------------------------

# Level 8 --- Calculus for Machine Learning

## Module 31 --- Calculus of Neural Networks

Take a computational graph:

\[ x `\rightarrow`{=tex} Wx+b `\rightarrow`{=tex} `\sigma`{=tex}
`\rightarrow`{=tex} Wx+b `\rightarrow`{=tex} `\hat`{=tex}y
`\rightarrow`{=tex} L \]

and understand its gradient structure.

Topics:

-   Computational graphs
-   Backpropagation
-   Chain rule at scale
-   Reverse-mode automatic differentiation
-   Forward-mode automatic differentiation
-   VJPs
-   JVPs
-   Jacobians
-   Batch gradients
-   Parameter gradients

Use PyTorch extensively.

------------------------------------------------------------------------

## Module 32 --- Calculus of Probability

Connect calculus with probability and statistics.

Topics:

-   Probability density functions
-   Cumulative distribution functions
-   Integrating densities
-   Expectations
-   Variance
-   Gaussian distributions
-   Multivariate Gaussians
-   Log likelihood
-   Maximum likelihood
-   Entropy
-   Cross entropy
-   KL divergence

------------------------------------------------------------------------

## Module 33 --- Optimization of Neural Networks

Study real loss landscapes.

\[ `\nabla`{=tex}\_`\theta`{=tex}L \]

and:

\[ H\_`\theta`{=tex}L \]

Topics:

-   SGD
-   Mini-batch SGD
-   Momentum
-   Adam
-   Curvature
-   Hessian eigenvalues
-   Saddle points
-   Exploding gradients
-   Vanishing gradients
-   Gradient clipping
-   Initialization
-   Optimization landscapes

------------------------------------------------------------------------

## Module 34 --- Calculus of Attention and Transformers

Study:

\[ Q=XW_Q,`\qquad`{=tex} K=XW_K,`\qquad`{=tex} V=XW_V \]

\[ A= `\operatorname{softmax}`{=tex} `\left`{=tex}(
`\frac{QK^\top}{\sqrt{d_k}}`{=tex} `\right`{=tex}) \]

\[ Y=AV \]

Investigate quantities such as:

\[ `\frac{\partial L}{\partial W_Q}`{=tex} \]

Topics:

-   Matrix derivatives
-   Softmax derivatives
-   Attention gradients
-   Gradient flow through attention
-   Parameter sensitivity
-   Jacobian structure
-   Backpropagation through Transformer components

The objective is to connect basic derivative intuition directly to
modern neural-network mathematics.

------------------------------------------------------------------------

# Complete Learning Path

``` text
FOUNDATIONS
│
├── Functions
│
├── Limits & Continuity
│
├── Derivatives
│
├── Differentials
│
├── Chain Rule
│
├── Taylor Approximations
│
├── Integration
│
└── Fundamental Theorem of Calculus
        │
        ▼
MULTIVARIABLE CALCULUS
│
├── Multivariable Functions
├── Partial Derivatives
├── Gradients
├── Directional Derivatives
├── Total Differentials
├── Jacobians
└── Hessians
        │
        ▼
OPTIMIZATION
│
├── Gradient Descent
├── Newton's Method
└── Constrained Optimization
        │
        ▼
VECTOR CALCULUS
│
├── Vector Fields
├── Multiple Integrals
├── Coordinate Transformations
├── Line Integrals
├── Surface Integrals
├── Divergence
├── Curl
├── Conservative Fields
├── Green's Theorem
├── Stokes' Theorem
└── Divergence Theorem
        │
        ▼
DYNAMICAL SYSTEMS
│
├── ODEs
├── Coupled ODEs
├── Dynamical Systems
└── PDEs
        │
        ▼
ML CALCULUS
│
├── Computational Graphs
├── Automatic Differentiation
├── Backpropagation
├── Probability + Calculus
├── Loss Functions
├── Neural Network Optimization
└── Attention / Transformers
```

------------------------------------------------------------------------

# Exercise Philosophy

Target approximately **600 exercises**.

These should **not** primarily be pencil-and-paper symbolic
manipulation.

A typical exercise should follow this loop:

1.  **Predict** what you think will happen.
2.  **Visualize** the function, surface, vector field, or trajectory.
3.  **Perturb** the input and observe the output change.
4.  **Compute** the mathematical quantity with Python.
5.  **Compare** numerical, symbolic, and autodiff approaches when
    useful.
6.  **Interpret** what the result means geometrically or physically.
7.  **Connect** the concept to later ideas such as optimization or
    machine learning.

## Example exercise pattern

Given:

\[ f(x)=x\^3 \]

At (x=2):

1.  Predict whether (f) is increasing or decreasing.
2.  Change (x) from (2) to (2.001).
3.  Measure the resulting change in (f(x)).
4.  Estimate the derivative using finite differences.
5.  Ask SymPy for the symbolic derivative.
6.  Ask PyTorch for the automatic derivative.
7.  Compare all three.
8.  Explain what (f'(2)=12) means in terms of a tiny perturbation of
    (x).

This pattern should gradually scale from (x\^3) all the way to neural
networks and attention.

------------------------------------------------------------------------

# Recommended Library Progression

## Early calculus

Use primarily:

-   NumPy
-   Matplotlib
-   SymPy
-   SciPy

Focus on numerical and visual intuition.

## Differential and multivariable calculus

Introduce PyTorch heavily:

-   `torch.autograd`
-   `torch.func.grad`
-   `torch.func.jacrev`
-   `torch.func.jacfwd`
-   `torch.func.hessian`

## Integration and differential equations

Use:

-   `scipy.integrate`
-   NumPy
-   Matplotlib
-   SymPy where symbolic solutions improve understanding

## Optimization

Use:

-   PyTorch
-   `torch.optim`
-   `scipy.optimize`

Implement simple algorithms manually before using library
implementations.

## Vector calculus

Use:

-   NumPy
-   SciPy
-   SymPy
-   Matplotlib 2D/3D visualization

## Machine-learning calculus

Use PyTorch as the primary environment.

------------------------------------------------------------------------

# Final Objective

By the end of the syllabus, notation such as

\[ `\nabla`{=tex}f,`\qquad`{=tex} J_f,`\qquad`{=tex} H_f,`\qquad`{=tex}
`\frac{\partial f}{\partial x}`{=tex},`\qquad`{=tex} `\int`{=tex}f(x),dx
\]

should represent **concrete computational and geometric ideas**, rather
than formulas to memorize.

The desired progression is:

\[ `\text{finite changes}`{=tex} `\rightarrow`{=tex}
`\text{limits}`{=tex} `\rightarrow`{=tex} `\text{derivatives}`{=tex}
`\rightarrow`{=tex} `\text{differentials}`{=tex} `\rightarrow`{=tex}
`\text{gradients}`{=tex} `\rightarrow`{=tex} `\text{Jacobians}`{=tex}
`\rightarrow`{=tex} `\text{optimization}`{=tex} `\rightarrow`{=tex}
`\text{dynamical systems}`{=tex} `\rightarrow`{=tex}
`\text{automatic differentiation}`{=tex} `\rightarrow`{=tex}
`\text{backpropagation}`{=tex} \]

The goal is not merely to know how a library calculates a derivative. It
is to develop enough intuition that when PyTorch returns a gradient, you
understand **what that gradient means, why its direction matters, how
perturbations propagate through the computation, and how that
mathematics drives modern machine learning.**
