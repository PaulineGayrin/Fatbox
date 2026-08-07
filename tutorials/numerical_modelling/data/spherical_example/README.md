# Spherical numerical-model teaching dataset

These archives contain the prepared fields required by the spherical Fatbox
tutorials. The workflow is independent of the numerical-model software or
original file format.

The bundled example data were derived from the surface output of a spherical
ASPECT model by Michael Pons; ASPECT is not required to run the tutorials.

Each compressed `.npz` file contains only:

- `strain`: a `180 x 360` surface strain-rate magnitude raster in `s^-1`;
- `velocity`: a `180 x 360 x 3` Earth-centred Cartesian surface-velocity
  raster in `m/year`;
- output number, model time and grid shape.

Strain is used for fault extraction and progressive correlation. Velocity is
used only by the slip-rate section of Tutorial 4. Meshes, cells, full tensors
and unrelated model fields are not included.

`manifest.json` records output times, units, preprocessing and SHA-256
checksums. The example data are provided under the Creative Commons
Attribution 4.0 International license (CC BY 4.0), with attribution to Michael
Pons.
