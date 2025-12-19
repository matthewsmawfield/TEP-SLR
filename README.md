# TEP-GL: Temporal-Spatial Coupling in Gravitational Lensing

Standard gravitational lensing analysis relies on the Isochrony Axiom—the implicit assumption that the observed image represents a synchronous spatial snapshot of the source. This work demonstrates that for evolving sources, this approximation breaks down in the presence of conformal metric couplings, creating a "temporal composite" image.

> **Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations**  
> TEP-GL Paper | v0.1 (Tortola) | DOI: 10.5281/zenodo.17982541

**Live Site**: [matthewsmawfield.github.io/TEP-GL](https://matthewsmawfield.github.io/TEP-GL/)

**The TEP Research Program:**
1. [**TEP Theory**](https://doi.org/10.5281/zenodo.16921911) (Foundational framework)
2. **TEP-GL** (This work - Gravitational Lensing Application)

When using this work, please cite the paper and theoretical framework listed below.

## Core Hypothesis

If the Isochrony Axiom is violated by differential time dilation (conformal metric coupling), extended images become temporal composites. This projects temporal depth onto the spatial plane, generating a Temporal Jacobian contribution that is mathematically indistinguishable from gravitational shear—a phenomenon defined here as Phantom Mass.

Crucially, GW170817 does not constrain this effect. Because photons and gravitational waves traverse the same null geodesics in the conformal limit, time dilation is common-mode and cancels in differential measurements. The constraints apply only to the disformal (cone-tilt) sector, leaving the conformal "rate of time" unconstrained.

## Key Predictions

1. **Variability Bias**: The inferred "dark matter" mass should correlate with the intrinsic variability of the source. Static sources should show less phantom mass than variable sources.
   
2. **Image Rotation (Non-Zero Curl)**: Unlike scalar-potential gravitational lensing (which is curl-free), the **Temporal Shear Tensor** possesses a non-zero curl, predicting unique image rotation effects.

3. **Chronometric Lensing**: Fast transients (FRBs) should exhibit millisecond-scale achromatic arrival-time residuals ("jitter") that cannot be explained by geometric time delays or plasma dispersion.

4. **Achromaticity**: Like dark matter, the temporal effect is wavelength-independent.

## Observational Discriminants

The manuscript proposes several tests to distinguish temporal-field effects from dark matter:

- **Source Evolution Analysis**: Compare apparent dark matter fraction vs source evolutionary timescale
- **Variable Source Monitoring**: Track morphological changes in strongly lensed AGN/quasars
- **Multi-Image Spectroscopy**: Search for evolutionary signatures between multiple images
- **Statistical Surveys**: Correlate lensing anomalies with source properties

## Theoretical Framework

This work builds on the Temporal Equivalence Principle (TEP), which proposes:
-   **Gravity is Geometry; Time is a Dynamical Field.**
-   The decomposition of proper time accumulation into "mass" and "time dilation" is **gauge-dependent**.
-   **Sector Decoupling**: The Conformal Sector (clock rates) is unconstrained by GW170817, while the Disformal Sector (speed of transmission) is tightly bound.

**TEP Theory Reference:**
> Smawfield, M. L. (2025). *Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed (v0.6 (Jakarta))*. Zenodo. DOI: [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911)

## File Structure

```
TEP-GL/
├── scripts/
│   ├── steps/                      # Analysis pipeline
│   └── utils/                      # Shared utilities
├── site/                           # Academic manuscript site
│   ├── components/                 # HTML section files
│   ├── public/                     # Static assets
│   └── dist/                       # Built site output
├── docs/                           # PDF versions
├── results/
│   ├── figures/                    # Generated plots
│   └── outputs/                    # Analysis results
├── logs/                           # Execution logs
├── manuscript-tep-gl.md            # Auto-generated markdown
└── VERSION.json                    # Version metadata
```

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- Astropy (for cosmological calculations)
- Lenstronomy (for lens modeling)

See `requirements.txt` for complete dependencies.

## Methodology

- **Lens Modeling**: Ray-tracing with temporal field variations
- **Source Evolution**: Myr-scale evolutionary models for galaxies/AGN
- **Time Delay Calculation**: Path-dependent proper-time accumulation
- **Image Reconstruction**: Temporal composite modeling
- **Statistical Analysis**: Correlation tests with observational data

## Related Work

- [TEP Theory](https://doi.org/10.5281/zenodo.16921911) - Foundational framework

## License

This project is licensed under Creative Commons Attribution 4.0 International (CC-BY-4.0). See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{smawfield2025tepgl,
  title={Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2025},
  doi={10.5281/zenodo.17982541},
  note={Preprint v0.1 (Tortola)}
}
```

## Acknowledgments

The author thanks colleagues for valuable discussions. This research made use of NASA's Astrophysics Data System and the arXiv preprint server.

## Status

**Version 0.1 (Tortola)** - Landmark Release. Theoretical framework complete.
