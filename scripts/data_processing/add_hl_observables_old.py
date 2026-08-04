"""Add vectorised high-level observables to an analysis Parquet shard.

This first version adds only per-jet kinematic quantities.  The input fields
have shape

    events x 2 jets

so NumPy ufuncs operate on every selected event and both jets without an
explicit Python loop.
"""

from __future__ import annotations

import awkward as ak
import numpy as np


JET_KINEMATIC_INPUTS = (
    "jet_px",
    "jet_py",
    "jet_pz",
)


def require_fields(data: ak.Array, required_fields: tuple[str, ...]) -> None:
    """Raise a clear error when an expected cache field is absent."""

    missing = [field for field in required_fields if field not in ak.fields(data)]
    if missing:
        raise ValueError(f"Input cache is missing required fields: {missing}")


def add_jet_kinematics(data: ak.Array) -> ak.Array:
    """Return the input records with vectorised per-jet kinematics added.

    Added fields
    ------------
    jet_pt
        Transverse momentum, sqrt(px**2 + py**2).
    jet_p
        Three-momentum magnitude, sqrt(px**2 + py**2 + pz**2).
    jet_phi
        Azimuthal angle in radians, in the interval [-pi, pi].
    jet_theta
        Polar angle from the +z axis in radians, in the interval [0, pi].
    """

    require_fields(data, JET_KINEMATIC_INPUTS)

    px = data.jet_px
    py = data.jet_py
    pz = data.jet_pz

    jet_pt = np.sqrt(px**2 + py**2)
    jet_p = np.sqrt(jet_pt**2 + pz**2)
    jet_phi = np.arctan2(py, px)
    jet_theta = np.arctan2(jet_pt, pz)

    enriched = ak.with_field(data, jet_pt, "jet_pt")
    enriched = ak.with_field(enriched, jet_p, "jet_p")
    enriched = ak.with_field(enriched, jet_phi, "jet_phi")
    enriched = ak.with_field(enriched, jet_theta, "jet_theta")

    return enriched

def add_event_kinematics(data):
    """Add vectorised observables constructed from the two selected jets."""

    required_fields = {
        "jet_energy",
        "jet_px",
        "jet_py",
        "jet_pz",
    }

    missing_fields = required_fields.difference(ak.fields(data))

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {sorted(missing_fields)}"
        )

    if not ak.all(ak.num(data.jet_energy, axis=1) == 2):
        raise ValueError("Every event must contain exactly two jets")

    # Extract leading and subleading jet components.
    e1 = data.jet_energy[:, 0]
    e2 = data.jet_energy[:, 1]

    px1 = data.jet_px[:, 0]
    px2 = data.jet_px[:, 1]

    py1 = data.jet_py[:, 0]
    py2 = data.jet_py[:, 1]

    pz1 = data.jet_pz[:, 0]
    pz2 = data.jet_pz[:, 1]

    # Difference between the two jet energies, normalised by total energy.
    energy_asymmetry = np.abs(e1 - e2) / (e1 + e2)

    # Transverse momentum of the two-jet system.
    dijet_px = px1 + px2
    dijet_py = py1 + py2
    dijet_pt = np.sqrt(dijet_px**2 + dijet_py**2)

    # Three-momentum magnitudes.
    p1 = np.sqrt(px1**2 + py1**2 + pz1**2)
    p2 = np.sqrt(px2**2 + py2**2 + pz2**2)

    # Opening angle between the jet momentum vectors.
    dot_product = px1 * px2 + py1 * py2 + pz1 * pz2
    cos_opening_angle = dot_product / (p1 * p2)

    # Protect arccos against small floating-point excursions.
    cos_opening_angle = ak.where(
        cos_opening_angle > 1.0,
        1.0,
        ak.where(cos_opening_angle < -1.0, -1.0, cos_opening_angle),
    )

    dijet_opening_angle = np.arccos(cos_opening_angle)
    dijet_acollinearity = np.pi - dijet_opening_angle

    enriched = ak.with_field(
        data, energy_asymmetry, "energy_asymmetry"
    )
    enriched = ak.with_field(
        enriched, dijet_pt, "dijet_pt"
    )
    enriched = ak.with_field(
        enriched, dijet_opening_angle, "dijet_opening_angle"
    )
    enriched = ak.with_field(
        enriched, dijet_acollinearity, "dijet_acollinearity"
    )

    return enriched

CONSTITUENT_ENERGY_INPUTS = (
    "constituent_energy",
    "constituent_multiplicity",
)


def add_constituent_energy_fractions(data):
    """Add the energy fraction of every constituent within its jet."""

    require_fields(data, CONSTITUENT_ENERGY_INPUTS)

    energy = data.constituent_energy

    # Shape: events × 2 jets
    total_constituent_energy = ak.sum(energy, axis=2)

    if not ak.all(total_constituent_energy > 0):
        raise ValueError(
            "At least one selected jet has non-positive total constituent energy"
        )

    # Add a length-one constituent axis so the denominator broadcasts
    # across every constituent in the corresponding jet.
    denominator = total_constituent_energy[:, :, np.newaxis]

    # Shape: events × 2 jets × constituents
    energy_fraction = energy / denominator

    enriched = ak.with_field(
        data,
        total_constituent_energy,
        "constituent_energy_sum",
    )
    enriched = ak.with_field(
        enriched,
        energy_fraction,
        "constituent_energy_fraction",
    )

    return enriched

E2_INPUTS = (
    "constituent_energy",
    "constituent_px",
    "constituent_py",
    "constituent_pz",
    "constituent_energy_fraction",
)


def add_e2(data, beta=0.2):
    """Add the two-point energy correlation function for each jet."""

    require_fields(data, E2_INPUTS)

    if beta <= 0:
        raise ValueError("beta must be positive")

    constituents = ak.zip(
        {
            "z": data.constituent_energy_fraction,
            "energy": data.constituent_energy,
            "px": data.constituent_px,
            "py": data.constituent_py,
            "pz": data.constituent_pz,
        }
    )

    # Construct every unique pair i < j within each jet.
    pairs = ak.combinations(
        constituents,
        2,
        axis=2,
        fields=["first", "second"],
    )

    first = pairs.first
    second = pairs.second

    minkowski_product = (
        first.energy * second.energy
        - first.px * second.px
        - first.py * second.py
        - first.pz * second.pz
    )

    energy_product = first.energy * second.energy

    # Avoid division by zero for any zero-energy constituent.
    safe_energy_product = ak.where(
        energy_product > 0,
        energy_product,
        1.0,
    )

    theta_squared = 2.0 * minkowski_product / safe_energy_product

    # Protect against small negative values from floating-point precision.
    theta_squared = np.maximum(theta_squared, 0.0)
    theta = np.sqrt(theta_squared)

    # A zero-energy constituent contributes z_i = 0.
    theta = ak.where(energy_product > 0, theta, 0.0)

    pair_contribution = (
        first.z
        * second.z
        * theta**beta
    )

    # Sum over all pairs, leaving one e2 value per jet.
    e2 = ak.sum(pair_contribution, axis=2)

    beta_name = str(beta).replace(".", "p")
    field_name = f"e2_beta_{beta_name}"

    return ak.with_field(data, e2, field_name)

def add_all_hl_observables(data, beta=0.2):
    """Add all currently validated high-level observables."""

    enriched = add_jet_kinematics(data)
    enriched = add_event_kinematics(enriched)
    enriched = add_constituent_energy_fractions(enriched)
    enriched = add_e2(enriched, beta=beta)

    return enriched

E3_INPUTS = (
    "constituent_energy",
    "constituent_px",
    "constituent_py",
    "constituent_pz",
    "constituent_energy_fraction",
)


def add_e3(data, beta=0.2):
    """Add the three-point energy correlation function for each jet."""

    require_fields(data, E3_INPUTS)

    if beta <= 0:
        raise ValueError("beta must be positive")

    constituents = ak.zip(
        {
            "z": data.constituent_energy_fraction,
            "energy": data.constituent_energy,
            "px": data.constituent_px,
            "py": data.constituent_py,
            "pz": data.constituent_pz,
        }
    )

    # Every unique constituent triple i < j < k.
    triples = ak.combinations(
        constituents,
        3,
        axis=2,
        fields=["first", "second", "third"],
    )

    first = triples.first
    second = triples.second
    third = triples.third

    def theta_between(a, b):
        """Paper's energy-normalised angular measure."""

        energy_product = a.energy * b.energy

        safe_energy_product = ak.where(
            energy_product > 0,
            energy_product,
            1.0,
        )

        lorentz_dot = (
            energy_product
            - a.px * b.px
            - a.py * b.py
            - a.pz * b.pz
        )

        theta_squared = (
            2.0 * lorentz_dot / safe_energy_product
        )

        # Numerical protection against small negative values.
        theta_squared = np.maximum(theta_squared, 0.0)
        theta = np.sqrt(theta_squared)

        # A non-positive-energy constituent has z = 0 and contributes nothing.
        return ak.where(energy_product > 0, theta, 0.0)

    theta_12 = theta_between(first, second)
    theta_13 = theta_between(first, third)
    theta_23 = theta_between(second, third)

    triple_contribution = (
        first.z
        * second.z
        * third.z
        * theta_12**beta
        * theta_13**beta
        * theta_23**beta
    )

    # Sum over triples, leaving one value for each selected jet.
    e3 = ak.sum(triple_contribution, axis=2)

    beta_name = str(beta).replace(".", "p")
    field_name = f"e3_beta_{beta_name}"

    return ak.with_field(data, e3, field_name)

def add_c2_d2(data, beta=0.2, minimum_e2=1e-12):
    """Add the C2 and D2 energy-correlation ratios for each jet."""

    if beta <= 0:
        raise ValueError("beta must be positive")

    if minimum_e2 <= 0:
        raise ValueError("minimum_e2 must be positive")

    beta_name = str(beta).replace(".", "p")

    e2_field = f"e2_beta_{beta_name}"
    e3_field = f"e3_beta_{beta_name}"
    c2_field = f"c2_beta_{beta_name}"
    d2_field = f"d2_beta_{beta_name}"

    require_fields(data, (e2_field, e3_field))

    e2 = data[e2_field]
    e3 = data[e3_field]

    # C2 and D2 are undefined when e2 is zero or extremely small.
    valid = (
        np.isfinite(e2)
        & np.isfinite(e3)
        & (e2 > minimum_e2)
    )

    # Prevent division by zero during array evaluation.
    safe_e2 = ak.where(valid, e2, 1.0)

    c2 = e3 / safe_e2**2
    d2 = e3 / safe_e2**3

    # Preserve invalid cases instead of assigning an arbitrary value.
    c2 = ak.where(valid, c2, np.nan)
    d2 = ak.where(valid, d2, np.nan)

    data = ak.with_field(data, c2, c2_field)
    data = ak.with_field(data, d2, d2_field)

    return data

def add_selected_jet_kinematics(data):
    """Add pT and three-momentum magnitude for the two selected jets."""

    require_fields(
        data,
        (
            "jet_px",
            "jet_py",
            "jet_pz",
        ),
    )

    jet_pt = np.sqrt(
        data.jet_px**2
        + data.jet_py**2
    )

    jet_p = np.sqrt(
        data.jet_px**2
        + data.jet_py**2
        + data.jet_pz**2
    )

    data = ak.with_field(data, jet_pt, "jet_pt")
    data = ak.with_field(data, jet_p, "jet_p")

    return data