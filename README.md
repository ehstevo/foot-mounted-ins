# GPS-INS-Boot - Foot-Mounted Pedestrian Inertial Navigation

A from-first-principles build of a **boot-mounted pedestrian inertial navigation system**: a
ZUPT-aided strapdown INS fused with GNSS through a **loosely-coupled error-state Extended Kalman
Filter**, estimating position, velocity, and attitude from a foot-worn IMU.

The goal is a correct, well-tested, research-grade implementation where every component is derived
and understood, not a black box. Each module is built one rung at a time: *concept → derivation →
implementation → a test that fails if the understanding is wrong.*

> **Status:** the front end is in place - BLE capture with replayable logging, a delta-form parser,
> device-frame characterization, a from-scratch rotation toolkit (quaternion/DCM/Euler), and a
> **working strapdown mechanization** validated end-to-end against a trajectory simulator to machine
> precision. Next: drift & error dynamics, then the estimation stack (ZUPT + error-state EKF + GNSS).

---

## Why this is interesting

A foot-mounted IMU is a uniquely powerful navigation platform: every footfall produces a brief
**zero-velocity** instant that can be exploited to bound the otherwise-unbounded drift of dead
reckoning. Combined with GNSS for absolute position, it yields meter-level pedestrian tracking that
keeps working through GNSS outages.

The hard parts, and the focus of this project, are the unglamorous ones that decide whether it
actually works: rigorous **reference-frame and sign conventions**, correct **delta-form strapdown
mechanization**, honest **error modeling**, robust **outlier and dropout handling**, and validation
against ground truth.

## Architecture (target)

```
IMU (BLE, 100 Hz, delta-form Δθ/Δv/Δt)
      │  replayable raw logging (every run reproducible)
      ▼
  strapdown mechanization  ──►  error-state EKF  ◄── ZUPT (stance detection)
      ▲                              ▲
      └──────── attitude/vel/pos ────┘  ◄── GNSS (loosely coupled, lever-arm + latency, NIS gating)
```

Key engineering choices:

- **Local-level NED** navigation frame; **quaternion** attitude (Hamilton, scalar-first, body→nav).
- **Delta-form mechanization** - consume coning/sculling-integrated increments directly at 100 Hz.
- **Error-state (indirect) EKF**, 15 states growing to include sensor biases and a sensor↔GNSS
  clock-offset term.
- **Source-agnostic ingest** - live BLE and replayed logs flow through one identical interface, so
  every result is reproducible offline.
- **Faithful parsing, separate judgement** - the parser decodes exactly what arrived; plausibility
  checks and dropout handling live in dedicated layers.

## Hardware

- Foot-worn **IMU module** (VectorNav VN-100 class) streaming delta-form increments over **BLE**.
- **u-blox GNSS** receiver wired to the compute board (PPS for timing).
- **Raspberry Pi** for capture, logging, and fusion.

> The module's wire-level interface is defined by a vendor ICD that is **not redistributable**, so
> device-specific identifiers and byte layout are not in this repo. They live in a local, untracked
> config (see Setup); the code is written against a generic delta-form IMU interface.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install bleak numpy scipy pyyaml pytest

# Device-specific identifiers are kept out of version control:
cp config/device.example.py config/device_local.py
# then fill config/device_local.py with your device's UUIDs
```

Bring-up (run on a Linux host with a Bluetooth adapter):

```bash
python scripts/ble_scan.py      # Rung 1: discover the module over BLE
```

## Roadmap

- [x] Reference-frame & sign-convention contract
- [x] BLE data-pipeline design (ingest → log → parse → quality/gap handling)
- [x] BLE capture + replayable raw logging
- [x] Delta-form parser + device-frame characterization
- [x] Rotation toolkit (quaternion/DCM/Euler) with analytic tests
- [x] Strapdown mechanization (validated on a trajectory simulator)
- [ ] Drift & error dynamics (linearized error propagation)
- [ ] ZUPT stance detection
- [ ] Error-state EKF + GNSS aiding
- [ ] Validation against RTK ground truth

## Development

This is a deliberate first-principles learning build. I implement and test the components myself,
using an AI assistant (Claude) as a design partner and mentor, for derivations, architecture
review, and catching mistakes, while the engineering decisions, and the understanding behind them,
are my own.

## License

TBD.
