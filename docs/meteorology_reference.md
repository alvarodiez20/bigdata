# Meteorology & Weather Prediction: A Technical Reference


## 1. The Mathematical Foundation of Numerical Weather Prediction (NWP)

### 1.1 What is NWP?

Numerical Weather Prediction is the art of integrating a system of partial differential equations that describe the evolution of the atmosphere forward in time, starting from an estimate of the current atmospheric state. Every major weather forecast you have ever seen — whether from ECMWF, NOAA, or AEMET — is ultimately the output of this process.

The atmosphere is a thin, stratified, rotating fluid shell on a sphere. Its governing equations are variants of the **Navier-Stokes equations**, augmented with thermodynamics, moisture physics, and the effects of Earth's rotation and spherical geometry.

### 1.2 The Primitive Equations

The standard set used in operational NWP is known as the **Primitive Equations** (PEs). They are obtained from the full Navier-Stokes equations by applying two approximations: the **hydrostatic approximation** (vertical accelerations are negligible compared to the gravitational force) and the **shallow-atmosphere approximation** (the depth of the atmosphere is small compared to Earth's radius $a$).

We work in a coordinate system where the horizontal coordinates are longitude $\lambda$ and latitude $\phi$, and the vertical coordinate is pressure $p$ (isobaric coordinates), which is more natural than geometric height because it automatically follows the mass of the atmosphere.

#### Momentum Equations (Horizontal)

$$\frac{Du}{Dt} - \left(f + \frac{u \tan\phi}{a}\right)v = -\frac{1}{a\cos\phi}\frac{\partial \Phi}{\partial \lambda} + F_\lambda$$

$$\frac{Dv}{Dt} + \left(f + \frac{u \tan\phi}{a}\right)u = -\frac{1}{a}\frac{\partial \Phi}{\partial \phi} + F_\phi$$

where:

- $u$, $v$ — zonal (eastward) and meridional (northward) wind components $[\text{m s}^{-1}]$
- $f = 2\Omega\sin\phi$ — Coriolis parameter, with $\Omega = 7.292 \times 10^{-5}\ \text{rad s}^{-1}$ Earth's angular velocity
- $\Phi = gz$ — geopotential $[\text{m}^2 \text{s}^{-2}]$, where $g$ is gravitational acceleration and $z$ is geometric height
- $F_\lambda, F_\phi$ — frictional/turbulent forcing terms
- $\frac{D}{Dt} = \frac{\partial}{\partial t} + \frac{u}{a\cos\phi}\frac{\partial}{\partial \lambda} + \frac{v}{a}\frac{\partial}{\partial \phi} + \dot{p}\frac{\partial}{\partial p}$ — material (Lagrangian) derivative in pressure coordinates

The Coriolis term $fv$ is what curves moving air masses into cyclonic and anticyclonic patterns — the term responsible for the large-scale rotating structure of weather systems.

#### Hydrostatic Equation

The hydrostatic approximation replaces the vertical momentum equation with the simpler balance:

$$\frac{\partial \Phi}{\partial p} = -\frac{RT}{p}$$

where $R = 287\ \text{J kg}^{-1} \text{K}^{-1}$ is the specific gas constant for dry air and $T$ is temperature. This diagnostic relationship means vertical motion $\dot{p} = Dp/Dt$ is not a prognostic variable but is diagnosed from horizontal divergence.

#### Thermodynamic Energy Equation

$$\frac{DT}{Dt} - \frac{RT\,\omega}{c_p\, p} = \frac{Q}{c_p}$$

where:

- $\omega = \dot{p} = Dp/Dt$ — vertical velocity in pressure coordinates $[\text{Pa s}^{-1}]$
- $c_p = 1004\ \text{J kg}^{-1}\text{K}^{-1}$ — specific heat of dry air at constant pressure
- $Q$ — diabatic heating rate per unit mass (radiation, latent heat release from condensation, turbulent heat flux) $[\text{W kg}^{-1}]$

The term $\frac{RT\,\omega}{c_p\, p}$ represents **adiabatic cooling/warming**: rising air (negative $\omega$ in pressure coordinates) cools as it expands.

#### Continuity Equation (Mass Conservation)

$$\frac{\partial u}{\partial \lambda}\frac{1}{a\cos\phi} + \frac{\partial(v\cos\phi)}{\partial \phi}\frac{1}{a\cos\phi} + \frac{\partial \omega}{\partial p} = 0$$

This states that the three-dimensional divergence of the wind is zero — what flows in horizontally must flow out vertically. This equation is used to diagnose $\omega$ from the horizontal wind field.

#### Equation of State

$$p = \rho R T$$

where $\rho$ is density $[\text{kg m}^{-3}]$. In pressure coordinates this is implicit.

#### Water Vapour Continuity

$$\frac{Dq}{Dt} = S_q$$

where $q$ is specific humidity $[\text{kg kg}^{-1}]$ and $S_q$ represents sources and sinks from condensation, evaporation, and precipitation. Modern NWP models carry additional prognostic variables for cloud liquid water, cloud ice, rain, snow, and graupel.

### 1.3 Discretisation and Numerical Methods

The continuous PDE system above cannot be solved analytically. Operational models discretise it using one of two approaches:

**Spectral methods (ECMWF IFS)**
The horizontal fields are expanded as truncated series of **spherical harmonics** $Y_n^m(\lambda, \phi)$:

$$u(\lambda, \phi, p, t) \approx \sum_{m=-M}^{M}\sum_{n=|m|}^{N} \hat{u}_n^m(p,t)\, Y_n^m(\lambda, \phi)$$

Derivatives in spectral space are exact multiplications. The ECMWF IFS at resolution T$_\text{co}$1279 retains $N \approx 1279$ wavenumbers, corresponding to a grid spacing of roughly 9 km.

**Finite-difference / finite-volume methods (GFS, WRF)**
Fields are stored on a discrete grid. Spatial derivatives are approximated by finite differences. Modern models use **semi-implicit, semi-Lagrangian** time-stepping to allow large time steps without numerical instability, exploiting the linearity of the gravity-wave terms.

### 1.4 Chaos and the Butterfly Effect

The primitive equations form a **nonlinear dynamical system**. Lorenz (1963) showed that even a dramatically simplified 3-variable system:

$$\dot{X} = \sigma(Y - X), \qquad \dot{Y} = X(\rho - Z) - Y, \qquad \dot{Z} = XY - \beta Z$$

exhibits **sensitive dependence on initial conditions**: trajectories starting from infinitesimally different points diverge exponentially at a rate characterised by the **largest Lyapunov exponent** $\lambda_1 > 0$. For the real atmosphere, $\lambda_1^{-1} \approx 2$–3 days, implying that errors double roughly every two days. This is the fundamental mathematical reason why **deterministic weather forecasts beyond ~10 days are impossible**, regardless of model resolution or computing power.

---

## 2. Observational Data and Ingestion

The initial condition — the state of the atmosphere at time $t_0$ — must be estimated from real observations. This is the **analysis** step, and its quality is the single largest source of forecast error.

### 2.1 Surface Weather Stations (SYNOP)

A global network of roughly 10,000 land-based stations (WMO SYNOP format) reporting every hour or three hours: temperature $T$, dewpoint $T_d$, pressure $p$ (reduced to mean sea level), wind speed and direction $(\vec{v})$, visibility, precipitation, and present weather. Spain's AEMET operates approximately 800 automatic weather stations (EMA network). Data are encoded in fixed-format or BUFR messages and disseminated over the WMO Global Telecommunication System (GTS).

### 2.2 Radiosondes

Balloon-borne instrument packages launched twice daily (00 UTC and 12 UTC) from roughly 800 stations worldwide. They measure vertical profiles of $T$, $T_d$, $p$, $\vec{v}$ from the surface up to ~30–35 km (stratopause). Each profile contains $\sim 200$–1000 levels of data. Radiosondes are the **primary in-situ source of upper-air information** and serve as the ground truth for validating satellite retrievals.

### 2.3 Aircraft (AMDAR/AIREP)

Commercial aircraft equipped with AMDAR (Aircraft Meteorological Data Relay) systems report $T$ and $\vec{v}$ automatically during ascent, cruise, and descent. Roughly 700,000–1,000,000 reports per day globally. Particularly dense over Europe and North America; sparse over oceans and the Southern Hemisphere — a critical data gap.

### 2.4 Ocean Buoys and Ships

- **Moored buoys**: Fixed position, continuous reporting of sea-surface temperature (SST), surface wind, pressure.
- **Drifting buoys** (Argo programme, ~4,000 active): Profiling floats that cycle between the surface and 2000 m depth, measuring $T$ and salinity.
- **Voluntary Observing Ships (VOS)**: Merchant vessels reporting surface marine observations.

### 2.5 Weather Radars

Ground-based Doppler radars operating at C-band or S-band transmit pulses and measure the backscattered power from precipitation particles. Key retrieved quantities:

- **Reflectivity** $Z$ $[\text{dBZ}]$: proxy for precipitation intensity via the $Z$–$R$ relationship $Z = aR^b$.
- **Radial velocity**: Doppler-shifted return gives the component of wind along the radar beam. A network of radars enables **dual-Doppler** wind retrieval.
- **Dual-polarisation** (modern systems): Co- and cross-polar returns distinguish rain, hail, snow, and biological targets.

Spatial resolution is typically 0.5–1 km at ranges up to 250 km. Weather radar data are critical for **nowcasting** (0–6 h).

### 2.6 Satellites

Satellites provide the most spatially uniform global coverage. Two orbital categories:

**Geostationary (GEO)**: Fixed over the equator at $\approx 36{,}000$ km altitude. Instruments: Meteosat (EUMETSAT), GOES (NOAA), Himawari (JMA). High temporal resolution (5–15 min), full-disc images. Provide cloud-top temperature (proxy for cloud height), water vapour imagery, and Atmospheric Motion Vectors (AMVs) derived from tracking cloud or water-vapour features between consecutive images.

**Low Earth Orbit (LEO)**: Polar or near-polar orbit at $\approx 700$–900 km, revisiting any point every ~12 h (or more frequently for constellations). Instruments include:

- **Passive microwave sounders** (e.g., AMSU, ATMS): Brightness temperatures in $O_2$ and $H_2O$ absorption bands, used to retrieve vertical profiles of $T$ and humidity.
- **Infrared sounders** (IASI, CrIS): Hyperspectral infrared spectrometers with thousands of channels, providing high-vertical-resolution $T$ and humidity profiles.
- **Scatterometers** (ASCAT): Measure radar backscatter from ocean surface roughness to retrieve near-surface wind speed and direction.
- **GNSS Radio Occultation** (COSMIC, Metop): GPS signals refracted by the atmosphere; the bending angle profile yields precise $T$ and humidity soundings with global coverage and no calibration drift.

The **assimilation of satellite radiances** — not retrieved geophysical variables but the raw brightness temperatures themselves — is the single largest contribution to modern forecast skill.

---

## 3. Data Assimilation: The Mathematical Bridge

Given a prior estimate of the atmospheric state (the **background**, $\mathbf{x}^b$, typically a short-range forecast) and a set of observations $\mathbf{y}^o$, data assimilation (DA) produces the **analysis** $\mathbf{x}^a$: the statistically optimal estimate of the true state $\mathbf{x}^t$.

The state vector $\mathbf{x} \in \mathbb{R}^n$ contains all model variables at all grid points. For a global NWP model, $n \sim 10^8$–$10^9$. The observation vector $\mathbf{y}^o \in \mathbb{R}^m$ may contain $m \sim 10^6$–$10^7$ observations per assimilation cycle (typically 6 or 12 hours).

### 3.1 Bayesian Framework

Assuming Gaussian errors, Bayes' theorem gives the maximum a posteriori (MAP) analysis as the minimiser of the **cost function**:

$$J(\mathbf{x}) = \frac{1}{2}(\mathbf{x} - \mathbf{x}^b)^\top \mathbf{B}^{-1}(\mathbf{x} - \mathbf{x}^b) + \frac{1}{2}(\mathbf{y}^o - H(\mathbf{x}))^\top \mathbf{R}^{-1}(\mathbf{y}^o - H(\mathbf{x}))$$

where:

- $\mathbf{B} \in \mathbb{R}^{n \times n}$ — **background error covariance matrix**: encodes how errors in $\mathbf{x}^b$ are spatially correlated. It is never stored explicitly (too large); instead, its action is approximated via spectral or wavelet transforms.
- $\mathbf{R} \in \mathbb{R}^{m \times m}$ — **observation error covariance matrix**: combines instrument error and representativity error (the mismatch between what the instrument measures and the corresponding model quantity).
- $H: \mathbb{R}^n \to \mathbb{R}^m$ — **observation operator**: maps from model space to observation space (e.g., a radiative transfer model that predicts satellite brightness temperatures from model $T$ and humidity profiles).

The gradient of $J$ is:

$$\nabla_\mathbf{x} J = \mathbf{B}^{-1}(\mathbf{x} - \mathbf{x}^b) - \mathbf{H}^\top \mathbf{R}^{-1}(\mathbf{y}^o - H(\mathbf{x}))$$

where $\mathbf{H} = \partial H / \partial \mathbf{x}$ is the tangent-linear observation operator (Jacobian).

### 3.2 3D-Var

In **3D-Var** (three-dimensional variational assimilation), $J$ is minimised over the model state at a single time level, ignoring the time dimension. The minimisation is performed iteratively using gradient-based methods (L-BFGS-B). 3D-Var is cheap but does not use the time evolution of observations within the assimilation window — every observation is treated as if it were valid at analysis time.

### 3.3 4D-Var

**4D-Var** extends the cost function over a time window $[t_0, t_N]$:

$$J(\mathbf{x}_0) = \frac{1}{2}(\mathbf{x}_0 - \mathbf{x}^b_0)^\top \mathbf{B}^{-1}(\mathbf{x}_0 - \mathbf{x}^b_0) + \frac{1}{2}\sum_{i=0}^{N}(\mathbf{y}^o_i - H_i(\mathbf{x}_i))^\top \mathbf{R}_i^{-1}(\mathbf{y}^o_i - H_i(\mathbf{x}_i))$$

where $\mathbf{x}_i = \mathcal{M}_{0 \to i}(\mathbf{x}_0)$ is the model state at time $t_i$ evolved from the control variable $\mathbf{x}_0$ using the nonlinear model $\mathcal{M}$.

Computing $\nabla_{\mathbf{x}_0} J$ requires the **adjoint model** $\mathcal{M}^\ast$, which propagates sensitivities backward in time:

$$\nabla_{\mathbf{x}_0} J = \mathbf{B}^{-1}(\mathbf{x}_0 - \mathbf{x}^b_0) + \sum_{i=0}^{N} \mathbf{M}_{0 \to i}^\top \mathbf{H}_i^\top \mathbf{R}_i^{-1}(\mathbf{y}^o_i - H_i(\mathbf{x}_i))$$

where $\mathbf{M}_{0 \to i}^\top$ is the adjoint of the tangent-linear model. Developing and maintaining the adjoint is a formidable software engineering challenge — ECMWF's 4D-Var system (operational since 1997) remains the gold standard of DA and is one of the primary reasons for ECMWF's consistent forecast superiority.

### 3.4 Ensemble Kalman Filter (EnKF)

The **Ensemble Kalman Filter** (Evensen 1994) is a Monte Carlo approximation of the Kalman filter equations. Instead of propagating the full error covariance $\mathbf{B}$ (which has $n^2 \sim 10^{16}$–$10^{18}$ elements), it is approximated by the **sample covariance** of an ensemble of $N_e$ model states $\{\mathbf{x}^{(k)}\}_{k=1}^{N_e}$:

$$\mathbf{P} \approx \frac{1}{N_e - 1}\sum_{k=1}^{N_e}(\mathbf{x}^{(k)} - \bar{\mathbf{x}})(\mathbf{x}^{(k)} - \bar{\mathbf{x}})^\top$$

The analysis update for each ensemble member is:

$$\mathbf{x}^{a(k)} = \mathbf{x}^{b(k)} + \mathbf{K}\left(\mathbf{y}^{o(k)} - H(\mathbf{x}^{b(k)})\right)$$

where $\mathbf{y}^{o(k)} = \mathbf{y}^o + \boldsymbol{\epsilon}^{(k)}$ is the observation perturbed with noise $\boldsymbol{\epsilon}^{(k)} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$ (the *perturbed observations* variant), and the **Kalman gain** matrix is:

$$\mathbf{K} = \mathbf{P}\mathbf{H}^\top(\mathbf{H}\mathbf{P}\mathbf{H}^\top + \mathbf{R})^{-1}$$

The ensemble-based $\mathbf{P}$ is **flow-dependent**: the error covariances change with the meteorological situation (different during, say, a blocking high versus an explosive cyclogenesis event). This is the key advantage over static-$\mathbf{B}$ methods. The Local Ensemble Transform Kalman Filter (LETKF) is widely used operationally due to its embarrassing parallelism.

Many operational centres now use **hybrid** methods that blend a static $\mathbf{B}$ with an ensemble-based $\mathbf{P}$, capturing the benefits of both.

---

## 4. Reanalysis Datasets

### 4.1 What is a Reanalysis?

A **reanalysis** is a retrospective application of a modern, fixed DA system to all available historical observations, producing a spatially complete, physically consistent, long-period gridded record of the atmospheric (and increasingly oceanic and land-surface) state. The two key words are:

- **Fixed system**: Unlike operational archives, the reanalysis uses a *single frozen model version* for the entire period. Operational archives suffer from discontinuities whenever the model or DA system is upgraded.
- **All available observations**: This includes observations that arrived too late for the operational forecast (delayed-mode data), quality-controlled historical ship logs, rescued paper records, etc.

### 4.2 ERA5: The Gold Standard

**ERA5** (ECMWF Re-Analysis 5th generation, Hersbach et al. 2020, [Copernicus Climate Change Service](https://climate.copernicus.eu)) is produced by ECMWF using a 2016 version of the IFS coupled to a land-surface model (HTESSEL) and a wave model. Key specifications:

| Property | Value |
|---|---|
| Horizontal resolution | 0.25° × 0.25° ($\approx$ 31 km) |
| Vertical levels | 137 model levels from surface to 80 km |
| Temporal resolution | Hourly |
| Period | 1940–present |
| Variables | $\sim$240 surface + pressure-level + single-level fields |
| Volume | $\sim$5 PB total |

ERA5 is derived using **10-member 4D-Var** with a 12-hour assimilation window, making the background error covariance flow-dependent. This gives it substantially better quality than its predecessor ERA-Interim, particularly in the Southern Hemisphere and upper troposphere.

### 4.3 Why ERA5 is Preferred for ML and Big Data Models

**Operational forecast archives** contain a mixture of model versions. A time series of, say, 850 hPa temperature extracted from ECMWF operational analyses will contain spurious jumps every time the model resolution was increased or the physics were updated. This *inhomogeneity* is devastating for statistical learning: the model will partially learn the model version as a feature.

ERA5 is preferred because:

1. **Temporal homogeneity**: A single model and DA system throughout. Trends and anomalies in ERA5 reflect actual climate signals, not model upgrades.
2. **Complete spatial coverage**: No missing values (by construction — the model fills gaps where observations are absent). This is essential for convolutional neural networks and spatiotemporal models that require complete grids.
3. **Long record**: 1940–present provides sufficient samples for training seasonal and climate-scale models.
4. **Uncertainty quantification**: The 10-member ensemble spread provides an estimate of analysis uncertainty at each grid point.
5. **Reproducibility**: The dataset is static (modulo minor corrections) and openly accessible via the CDS API.

**Caveats**: ERA5 is not observations — it is a model-filtered, smoothed representation. The effective resolution is coarser than the 31 km grid (approximately 2–3 × grid spacing due to spectral smoothing). Sub-grid phenomena (convection, fog, mountain-valley winds) are parameterised and may be biased. Always validate ERA5-derived statistics against independent station observations in your study region.

---

## 5. Global vs. Regional Models

### 5.1 The Global Titans

#### ECMWF IFS (Integrated Forecasting System)

| Property | Value |
|---|---|
| Centre | ECMWF, Reading, UK (intergovernmental consortium) |
| Core dynamics | Spectral, semi-Lagrangian, semi-implicit |
| Resolution (operational 2024) | T$_\text{co}$1279 $\approx$ 9 km, 137 levels |
| Ensemble (ENS) | 51 members at T$_\text{co}$639 ($\approx$ 18 km) |
| DA system | 4D-Var (12-h window) + ensemble |
| Forecast range | Deterministic 10 days; ENS 15 days; extended range 46 days |
| Renowned for | Upper-air accuracy; tropical cyclone tracks; European medium-range |

#### NOAA GFS (Global Forecast System)

| Property | Value |
|---|---|
| Centre | NCEP/EMC, NOAA, USA |
| Core dynamics | Finite-volume cubed-sphere (FV3) since 2019 |
| Resolution (operational 2024) | C768 $\approx$ 13 km, 127 levels |
| Ensemble (GEFS) | 31 members at C384 ($\approx$ 25 km) |
| DA system | Hybrid EnKF–3D-Var (GSI) |
| Forecast range | Deterministic 16 days; GEFS 35 days |
| Renowned for | Open data policy (all output free); rapid updates (every 6 h); US precipitation |

**Historical performance**: Objective verification scores (e.g., anomaly correlation coefficient of 500 hPa geopotential) consistently show ECMWF leading GFS by approximately 12–18 hours in effective forecast skill. The gap has narrowed since GFS adopted FV3, but ECMWF remains the benchmark. The European centre's advantage stems primarily from its superior 4D-Var DA system and more sophisticated convection parameterisation.

### 5.2 Mesoscale / Regional Models

Global models cannot resolve phenomena below $\sim$20–30 km: sea-breeze circulations, orographic precipitation enhancement, urban heat islands, thunderstorm cells. **Limited-area models (LAMs)** run at higher resolution over a domain of interest, **nested** inside a global model that provides boundary conditions.

#### WRF (Weather Research and Forecasting Model)

Open-source, community model developed by NCAR/NOAA. Finite-difference dynamics on an Arakawa-C staggered grid. Widely used for research and many operational applications worldwide. Typical research configurations: 1–4 km horizontal spacing, explicit convection (no parameterisation needed below $\sim$4 km). WRF domains can be nested down to $\mathcal{O}$(100 m) for wind energy or urban microclimate applications.

#### HARMONIE-AROME (ALADIN-HIRLAM system)

Convection-permitting model jointly developed by Météo-France and a consortium of European NWP centres (HIRLAM: Nordic countries, Ireland, Spain, Netherlands). Run operationally by **AEMET** over the Iberian Peninsula at **2.5 km horizontal resolution** with 3-hourly updates (HARMONIE-AROME cy43). Key features:

- Non-hydrostatic dynamics: necessary at convection-permitting scales where vertical accelerations are no longer negligible.
- Explicit deep convection: no Kain-Fritsch parameterisation; convective cells are resolved.
- Boundary conditions from the ECMWF IFS: the global ECMWF deterministic run feeds the lateral boundary conditions every 3 hours.
- Focus on high-impact weather: heavy precipitation, strong winds, fog, snow events over complex terrain (Pyrenees, Sistema Central, Sierra Nevada).

The nesting hierarchy is therefore:

$$\underbrace{\text{ECMWF IFS}}_{\approx 9\text{ km, global}} \longrightarrow \underbrace{\text{HARMONIE-AROME}}_{\approx 2.5\text{ km, Iberian Peninsula}} \longrightarrow \underbrace{\text{WRF/custom LAM}}_{\mathcal{O}(100\text{ m–1 km, local})}$$

---

## 6. Deterministic vs. Ensemble Forecasting

### 6.1 The Deterministic Paradigm

A **deterministic forecast** integrates the model equations from a single, best-estimate initial condition. The output is a single trajectory in phase space. Because the atmosphere is chaotic (Section 1.4), this trajectory diverges exponentially from the true trajectory, and the forecast becomes climatologically indistinguishable from a random state at approximately 2 weeks.

### 6.2 Ensemble Prediction Systems (EPS)

An **ensemble** runs $N_e$ forecasts from slightly different initial conditions and/or different model configurations, sampling the uncertainty in both the initial state and the model itself.

**Initial condition perturbations** are generated by:

- **Singular vectors (ECMWF ENS)**: Find the directions in state space that amplify most over the forecast period. These are the leading singular vectors of the tangent-linear model $\mathbf{M}$:
  $$\mathbf{M}^\top \mathbf{M}\, \mathbf{v}_k = \sigma_k^2\, \mathbf{v}_k$$
  Perturbations along the leading $\mathbf{v}_k$ (largest $\sigma_k$) capture the fastest-growing errors.

- **Ensemble Data Assimilation (EDA, ECMWF)**: Run the full DA system in ensemble mode; the spread of the analysis ensemble provides statistically consistent initial perturbations.

- **Breeding of Growing Modes (NCEP/GEFS)**: Iteratively amplify and rescale random perturbations to isolate the fastest-growing structures.

**Stochastic model perturbations** address model uncertainty:

- **SPPT (Stochastically Perturbed Parameterisation Tendencies)**: Multiply parameterised tendencies (convection, boundary layer, radiation) by a spatiotemporally correlated random field $r(\mathbf{x},t)$, $\langle r \rangle = 1$.
- **SKEB (Stochastic Kinetic Energy Backscatter)**: Inject random streamfunction perturbations to represent energy backscatter from unresolved scales.

### 6.3 Probabilistic Output

The ensemble produces a probability distribution over future states. For a scalar quantity $X$ (e.g., 24-h precipitation at a point), the empirical distribution is:

$$\hat{F}(x) = \frac{1}{N_e}\sum_{k=1}^{N_e}\mathbf{1}[X^{(k)} \leq x]$$

Standard derived products include:

- **Probability of exceedance**: $P(X > x_0) = 1 - \hat{F}(x_0)$. E.g., probability of precipitation exceeding 50 mm in 24 hours.
- **Ensemble mean**: $\bar{X} = N_e^{-1}\sum_k X^{(k)}$, smoother than any individual member (variance-cancellation).
- **Ensemble spread** $\sigma_\text{ens}$: Should equal the root-mean-square error of the ensemble mean if the ensemble is **reliable** (spread–skill relationship).
- **Spaghetti plots**: Contours of a single variable for all members simultaneously.

### 6.4 Reliability and Resolution

A probabilistic forecast is **reliable** (calibrated) if the stated probability $p$ of an event matches its long-run frequency of occurrence. Reliability is diagnosed via the **reliability diagram** (calibration curve) and the **rank histogram** (Talagrand diagram). A perfectly reliable ensemble has a flat rank histogram.

**Resolution** measures how much the forecast probabilities deviate from climatology — it is possible to be reliable but uninformative (always forecast climatological probability). A proper scoring rule like the **Continuous Ranked Probability Score (CRPS)**:

$$\text{CRPS}(F, x_o) = \int_{-\infty}^{\infty}(F(x) - \mathbf{1}[x \geq x_o])^2 \, dx$$

jointly measures reliability and resolution; lower is better. The deterministic RMSE is a special case.

---

## 7. Prediction Horizons

### 7.1 Nowcasting (0–6 hours)

At the sub-hourly to 6-hour range, **extrapolation-based** methods compete with and often outperform NWP. The dominant technique is **Lagrangian extrapolation**: estimate the current motion field of precipitation from a sequence of radar composites, then advect the current reflectivity field forward in time.

Mathematically, the radar reflectivity $Z(\mathbf{x}, t)$ is assumed to satisfy a transport equation:

$$\frac{\partial Z}{\partial t} + \mathbf{v} \cdot \nabla Z = S$$

where $\mathbf{v}$ is the motion field (estimated by optical flow, e.g., Lucas-Kanade or a variational method) and $S$ accounts for growth and decay. State-of-the-art nowcasting systems (PySTEPS, DeepMind NowcastNet) treat $S$ stochastically, generating ensemble nowcasts with physically realistic small-scale variability.

**Why NWP struggles at 0–2 hours**: The model requires time to spin up (generate its own precipitation structures from the smooth initial analysis), and the analysis increment from DA is not sufficient to properly initialise rapidly evolving convective systems.

### 7.2 Short-Range (6 h–3 days) and Medium-Range (3–10 days)

This is the sweet spot for NWP. Skill degrades smoothly following the Lorenz error-growth curve:

$$e(t) \approx e_0 e^{\lambda_1 t} \left(1 - \frac{e_0^2}{e_\infty^2}e^{2\lambda_1 t}\right)^{-1/2}$$

where $e_0$ is initial error, $e_\infty$ is the saturation (climatological) error level. The 500 hPa anomaly correlation coefficient (ACC) drops below 0.6 (the conventional useful-skill threshold) at approximately day 6–7 in extratropics. Modern ensemble systems push useful probabilistic information to ~14 days.

### 7.3 Subseasonal-to-Seasonal (S2S, 2 weeks–1 year)

Beyond ~2 weeks, atmospheric predictability from initial conditions is exhausted (the Lorenz barrier). Predictability at these scales stems from **slowly varying boundary conditions**:

- **Sea-surface temperatures (SSTs)**: El Niño–Southern Oscillation (ENSO) has a predictability of ~6–12 months and exerts strong global teleconnections.
- **Soil moisture and snow cover**: Integrators of surface energy balance with memory of weeks to months.
- **Stratospheric dynamics**: Sudden Stratospheric Warming events (SSWs) couple downward to the troposphere over 2–6 weeks, creating regime predictability windows.
- **Madden-Julian Oscillation (MJO)**: Dominant mode of tropical intraseasonal variability, predictable to ~4 weeks.

S2S forecasts are inherently probabilistic. Skill is expressed as anomaly correlation relative to a climatological baseline, often using hindcast (reforecast) statistics for calibration.

### 7.4 Climate Projections (Decadal and beyond)

Beyond seasonal timescales, the question shifts from *weather* to *climate*. Predictability derives entirely from external forcings (greenhouse gas concentrations, volcanic eruptions, solar variability) and internal low-frequency ocean modes. This is the domain of **Earth System Models (ESMs)** and is outside the scope of this course.

---

## 8. Big Data Formats in Meteorology

### 8.1 GRIB / GRIB2

**GRIB** (General Regularly-distributed Information in Binary form) is the WMO standard binary format for encoding gridded meteorological data. Virtually all NWP output from ECMWF, NOAA, and national agencies is distributed in GRIB2.

Key features:

- **Self-describing**: each GRIB message contains a header describing the variable (temperature, wind, geopotential), level (pressure level, height above ground, surface), time (reference time, validity time), and grid definition (Gaussian, regular lat–lon, Lambert conformal, etc.).
- **Packing**: values are packed as scaled integers; different packing methods (simple, complex, JPEG2000, PNG) provide compression ratios of 3:1 to 10:1.
- **Flat structure**: a GRIB file is a concatenation of independent GRIB messages; there is no file-level index. Efficient access requires an external index (via `eccodes`, `cfgrib`, or `wgrib2`).

Reading in Python:

```python
import cfgrib
import xarray as xr

ds = xr.open_dataset("gfs.t00z.pgrb2.0p25.f024",
                      engine="cfgrib",
                      filter_by_keys={"typeOfLevel": "isobaricInhPa",
                                      "shortName": "t"})
```

### 8.2 NetCDF (Network Common Data Form)

**NetCDF** (Unidata/UCAR) is the dominant format for scientific gridded data, reanalyses (ERA5, NCEP/NCAR), and model output archives. Version 4 uses HDF5 as the underlying storage format.

Key features:

- **Dimensions, variables, and attributes**: a NetCDF file is a self-describing dataset with named dimensions (e.g., `time`, `latitude`, `longitude`, `level`), typed variables (e.g., `float32 temperature[time, level, latitude, longitude]`), and metadata attributes (units, long name, fill value, coordinate reference system).
- **CF Conventions**: the Climate and Forecast (CF) metadata conventions define standard names, units, and coordinate systems, enabling generic tools to interpret any conforming file.
- **Chunking and compression**: HDF5 supports internal chunking (dividing the array into tiles for random access) and DEFLATE compression. Choosing the right chunk shape is critical for performance: chunk along the access pattern.
- **Random access**: unlike GRIB, any hyperslice of a NetCDF variable can be read without scanning the entire file.

Reading in Python:

```python
import xarray as xr

ds = xr.open_dataset("era5_temperature_2023.nc")
t850 = ds["t"].sel(level=850, method="nearest")  # lazy: no I/O yet
t850_mean = t850.mean(dim="time").compute()        # triggers I/O
```

### 8.3 Zarr

**Zarr** is a modern, cloud-native, chunked array storage format. It addresses the fundamental limitation of NetCDF for cloud workflows: NetCDF's HDF5 format requires efficient byte-range access, which is poorly supported by object stores (S3, GCS, Azure Blob).

Key features:

- **Chunk-based**: arrays are stored as independent files (or objects), one per chunk. Any chunk can be read without touching others — ideal for parallel access.
- **Multiple storage backends**: local filesystem, S3/GCS/Azure, ZIP archives, memory.
- **Compressors**: blosc, lz4, zstd, zlib — often achieving 5–10× compression on meteorological data.
- **No file-level lock**: multiple processes/workers can read (and write) simultaneously without contention.
- **ARCO-ERA5**: the ERA5 dataset has been re-encoded in Zarr format (Analysis-Ready Cloud-Optimised ERA5) and stored on Google Cloud Storage, enabling training of global ML weather models at scale without local data downloads.

```python
import xarray as xr
import zarr

# Open ARCO-ERA5 directly from GCS (no local download)
ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks={"time": 48}
)
```

### 8.4 Comparison

| | GRIB2 | NetCDF4 | Zarr |
|---|---|---|---|
| **Standard body** | WMO | Unidata/UCAR | Community / OGC |
| **Primary use** | Operational NWP distribution | Scientific archives, reanalyses | Cloud ML training, analysis |
| **Cloud access** | Poor (sequential scan) | Moderate (byte-range) | Excellent (native) |
| **Parallelism** | Limited | Moderate (read-only) | Full (read/write) |
| **Self-description** | Good (GRIB2 sections) | Excellent (CF conventions) | Good (JSON metadata) |
| **Python library** | `cfgrib`, `eccodes` | `netCDF4`, `xarray` | `zarr`, `xarray` |

---

## 9. Common Visualizations

### 9.1 Synoptic Charts and Isobar Maps

The classical tool of operational meteorology. Contours of mean sea level pressure (MSLP) plotted every 4 or 5 hPa reveal the large-scale flow pattern: **high-pressure anticyclones** (divergent, subsiding, fair weather), **low-pressure cyclones** (convergent, ascending, cloud and precipitation), and the **fronts** separating air masses of different origin and temperature.

In Python: `matplotlib.pyplot.contour` or `cartopy` with `ax.contour(lons, lats, mslp, levels=np.arange(960, 1040, 4), transform=ccrs.PlateCarree())`.

### 9.2 Spatial Heatmaps (Filled Contours)

`contourf` or `pcolormesh` maps visualise continuous scalar fields: 2-m temperature anomaly, 24-h precipitation accumulation, sea-surface temperature, wind speed. Diverging colourmaps (e.g., RdBu_r) are appropriate for anomalies; sequential colourmaps for absolute magnitudes. Always label units and include a colorbar with tick marks at physically meaningful values.

### 9.3 Wind Barbs and Vectors

A **wind barb** encodes both speed and direction in a single glyph: the staff points into the wind; pennants (50 kt), full barbs (10 kt), and half barbs (5 kt) sum to give the total speed. Standard on synoptic charts and upper-air analyses.

**Streamlines** and **quiver plots** (`matplotlib.pyplot.streamplot`, `.quiver`) visualise the wind vector field. For large grids, subsample to avoid visual clutter.

### 9.4 Meteograms

A **meteogram** is a time-series panel plot for a single location showing the evolution of multiple variables over the forecast period: temperature, dewpoint, precipitation (bar chart), wind (barbs on a time axis), cloud cover, MSLP. The ECMWF's ENS meteogram overlays the deterministic forecast (thick line), ensemble plumes (shaded percentile bands), and climatological quartiles for reference.

In Python: `matplotlib` figure with multiple `Axes` sharing the time axis (`sharex=True`), or `plotly` for interactive versions.

### 9.5 Skew-T Log-P Diagrams

The canonical thermodynamic diagram for radiosonde data. Temperature $T$ and dewpoint $T_d$ are plotted against $\log p$ on the vertical axis, with the temperature axis skewed 45° to the right to separate the two curves visually. Overlaid on the diagram are:

- **Dry adiabats**: lines of constant potential temperature $\theta = T(p_0/p)^{R/c_p}$.
- **Saturated adiabats**: lines followed by a parcel lifted past its lifting condensation level (LCL).
- **Mixing ratio lines**: constant $q$.

Key derived instability indices are read off geometrically: **CAPE** (Convective Available Potential Energy, the positive area between the parcel and environment temperature curves above the LFC), **CIN** (Convective Inhibition, the negative area below), **LCL**, **LFC** (Level of Free Convection), **EL** (Equilibrium Level):

$$\text{CAPE} = g \int_{\text{LFC}}^{\text{EL}} \frac{T_{\text{parcel}} - T_{\text{env}}}{T_{\text{env}}} \, dz$$

In Python: the `MetPy` library (`metpy.plots.SkewT`) renders Skew-T diagrams directly from `xarray` datasets.

### 9.6 Hovmöller Diagrams

A **Hovmöller diagram** plots a meteorological variable as a function of longitude (x-axis) and time (y-axis), averaged over a latitude band. Eastward-tilting coherent structures reveal **Rossby wave propagation**; westward tilt indicates **easterly waves** (tropical). Essential for diagnosing the MJO and for visualising the evolution of precipitation anomalies along a latitude band.

### 9.7 Spaghetti Plots and Ensemble Fan Charts

For ensemble output, **spaghetti plots** draw a single contour (e.g., the 1000 hPa geopotential height $= 500$ m contour) for every ensemble member simultaneously. Tight spaghetti = high confidence; widely spread spaghetti = high uncertainty.

**Fan charts** (probability shading) show the ensemble distribution of a 1D time series: the central 10%, 25%, 50%, 75%, and 90% probability envelopes are shaded in decreasing opacity around the ensemble mean.

### 9.8 Practical Python Stack

| Task | Library |
|---|---|
| Data I/O | `xarray`, `cfgrib`, `zarr`, `netCDF4` |
| Meteorological calculations | `MetPy` |
| Static maps | `cartopy` + `matplotlib` |
| Interactive maps | `hvplot`, `plotly`, `folium`, `leafmap` |
| Large-scale computation | `dask` (lazy arrays), `cupy` (GPU) |
| Statistical analysis | `scipy.stats`, `statsmodels`, `scikit-learn` |

---

## Further Reading

- **Holton & Hakim** — *An Introduction to Dynamic Meteorology* (5th ed.): rigorous treatment of the primitive equations and atmospheric dynamics.
- **Kalnay** — *Atmospheric Modeling, Data Assimilation and Predictability*: the standard graduate text on NWP and DA.
- **Evensen** — *Data Assimilation: The Ensemble Kalman Filter* (2nd ed.): comprehensive treatment of EnKF theory and practice.
- **Bauer, Thorpe & Brunet** (2015) — *The quiet revolution of numerical weather prediction*, *Nature* 525, 47–55: an accessible overview of modern NWP for a mathematical audience.
- **Hersbach et al.** (2020) — *The ERA5 global reanalysis*, *QJRMS* 146, 1999–2049: the primary reference for ERA5.
- **Rasp et al.** (2024) — *WeatherBench 2*: benchmark for ML weather models against ECMWF.
