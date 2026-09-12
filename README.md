# Lagrangian approach to origami vertex analysis: Multistability

Code and supporting data for:

**Matthew Grasinger, Andrew Gillman, and Philip R. Buskohl,  
“Lagrangian approach to origami vertex analysis: Multistability.”**

This repository contains the symbolic/numerical calculations and supporting
data used to generate the results in the manuscript.

## Overview

The analysis is divided into two main parts.

1. **Symmetry-reduced origami kinematics and energy landscapes**

   Wolfram Language (Mathematica) notebooks derive and evaluate the
   symmetry-reduced kinematic solutions used in the manuscript. These
   calculations generate compatible folded configurations within the imposed
   symmetry classes and account for self-contact within the
   symmetry-preserving configuration space.

   The resulting kinematic data are also exported to CSV files for use in
   subsequent analysis and figure generation.

2. **Full-space stability analysis**

   Python code tests the local stability of candidate minima after relaxing
   the imposed folding symmetry. The candidate states identified in the
   symmetry-reduced calculations are included directly in the degree-6 and
   degree-8 batch-analysis scripts.

   For each candidate, the Python analysis:
   - reconstructs the full fold-angle state;
   - computes the deformed crease directions;
   - constructs the linearized loop-closure operator;
   - identifies the tangent space of compatible fold-angle perturbations;
   - retracts finite perturbations onto exact nonlinear loop closure; and
   - evaluates the local gradient and Hessian of the crease energy on the
     full compatible configuration manifold.

   The calculation is repeated at multiple finite-difference step sizes to
   assess numerical robustness.

Self-contact is not imposed during this auxiliary full-space stability test.
Accordingly, states that remain strict minima when facet penetration is
permitted are certified as full-space stable, whereas destabilizing modes
that may be blocked by contact are treated conservatively in the manuscript.

## Repository contents

### Wolfram Language / Mathematica

- `origami-with-symmetry.nb`  
  Derivation and evaluation of the symmetry-reduced origami kinematics and
  associated energy landscapes.

- `origami-with-contact.nb`  
  Calculations associated with admissibility and self-contact in the
  symmetry-preserving configuration space.

- `*.csv`  
  Exported kinematic and/or candidate-state data used in the analysis and
  figure generation.

### Python

- `origami_fullspace_stability.py`  
  Core routines for the full-space local stability calculation, including
  loop-closure kinematics, tangent-space construction, nonlinear compatibility
  retraction, and numerical evaluation of the restricted gradient and Hessian.

- `degree6_stability_batch.py`  
  Batch full-space stability analysis for the degree-6 candidate states used
  in the manuscript. The candidate configurations analyzed in Figures 3 are
  included in this script.

- `degree8_stability_batch.py`  
  Batch full-space stability analysis for the degree-8 candidate states used
  in the manuscript. The candidate configurations analyzed in Figure 4 are
  included in this script.

- `print_fullspace_stable_states.py`  
  Post-processing utility that reads the batch-analysis output and identifies
  the states certified as stable in the full compatible configuration space.

## Typical workflow

### 1. Generate symmetry-reduced kinematics

Run the relevant Mathematica notebooks to derive/evaluate the
symmetry-preserving kinematics, determine admissible configurations, and
generate the candidate minima used in the manuscript.

The corresponding data may also be read directly from the supplied CSV files.

### 2. Run the degree-6 full-space stability analysis

```bash
python degree6_stability_batch.py
