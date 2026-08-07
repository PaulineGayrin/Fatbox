# Changelog

This file records user-visible changes to Fatbox. The spherical contribution
below is not yet released.

## Unreleased — spherical fault-network workflow

Contribution prepared: 7 August 2026

Contributor: Michael Pons

### Data

- Added eleven lightweight spherical surface archives for outputs 220–230.
- The example data were derived from surface output of a spherical ASPECT
  numerical model by Michael Pons.
- Each archive contains only the fields required by the tutorials: a
  `180 × 360` strain-rate magnitude raster, a three-component surface-velocity
  raster, output number, model time and grid shape.
- Strain rate is used for extraction and correlation. Velocity is retained
  only for the slip-rate exercise.
- Numerical-model meshes, cells, full tensors and unrelated fields are not
  distributed.

### Added

- A focused `spherical.py` numerical core for longitude/latitude conversion,
  great-circle distances, bearings, dateline-safe graph operations, trace
  direction, sampling, fault correlation and spherical slip calculations.
- Progressive fault tracking over arbitrary output sequences, including
  persistent identities, candidate families, split/merge information and
  recovery from a missing match by looking back to `n−2`.
- A dataset-independent `spherical_surface.py` helper containing only the
  prepared-archive loader, periodic graph extraction and raster-sampling
  operations required by the tutorials.
- Tutorial 3 for step-by-step spherical extraction followed by batch
  extraction over the complete example sequence.
- Tutorial 4 for one-pair correlation, progressive sequence correlation and
  interval-by-interval slip-rate calculation.
- Unit, integration and tutorial-data tests, Sphinx API documentation, and a
  reproducible spherical tutorial environment.

### Changed and corrected

- Spherical calculations use geographic node positions in
  `(longitude, latitude)` degrees and great-circle distances in kilometres.
- Graph connections and extraction are periodic across the ±180° dateline.
- Southern Hemisphere data are processed by default; geographic filtering is
  now an explicit user choice.
- Input-specific numerical-model processing is kept outside the spherical
  numerical core, allowing the tutorials to use similarly prepared data from
  ASPECT or another modelling system.

### Verification

- Both spherical notebooks execute from clean kernels and retain their
  teaching figures without machine-specific paths.
- The automated spherical test suite and strict Sphinx documentation build
  pass locally.
