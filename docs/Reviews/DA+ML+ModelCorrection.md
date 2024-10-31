# A Review of Data Assimilation and Machine Learning Approaches for Model Error Correction

## Introduction
Physics-based models often suffer from imperfections, primarily due to unresolved small-scale processes. This limitation significantly impacts the accuracy of weather forecasting systems.

## Data Assimilation Overview
Data assimilation addresses model errors by incorporating observational data, which inherently contains information about unresolved physics. While traditional methods like 3D-Var or 4D-Var focus on finding analysis fields directly, this paper explores approaches to correct inherent model errors using observational data.

## Model Correction Methods
The paper discusses two primary correction methods for dynamical systems of the form:

$$\frac{du}{dt} = F(u)$$

### 1. Resolvent Correction (RC)
- Focuses on correcting the discretized time stepper (resolvent)
- Example: For forward Euler discretization, resolvent is:
  $M(u^k) = u^k + \Delta t F(u^k)$
- Aims to improve the accuracy of $M$

### 2. Tendency Correction (TC)
- Targets improvement of the tendency function $F$ itself
- Represents a data-driven approach to identify unknown physics
- Both RC and TC can be viewed as methods for data-driven physics identification

## Implementation Approach
### Cost Function Minimization
- Uses carefully chosen cost functions
- Incorporates 4D-Var cost function (considered state-of-the-art)
- Process involves:
    1. Assimilation with correction model
    2. Improvement of corrected model using gradient descent
    3. Iteration in an alternating fashion (similar to coordinate gradient descent)

## Offline vs Online Methods
### Offline Method Limitations
- Requires complete procedure repetition for new observations
- Can be computationally intensive and time-consuming

### Online Method Advantages
- Eliminates need for complete recomputation with new data
- More efficient for continuous data streams

Note: Original paper available at [Journal of Computational Science.](https://www.sciencedirect.com/science/articles/pii/S1877750321001435)