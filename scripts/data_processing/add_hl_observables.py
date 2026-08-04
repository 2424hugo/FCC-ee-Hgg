"""Calculate the selected per-jet high-level observables.

The input is expected to contain exactly two selected jets per event and the
constituent four-vectors for each jet.  The returned array contains only:

    e2, e3, jet_pt, jet_p, c2, d2, jet_theta

Every output field has shape ``events x 2 jets``.  Jet index 0 is assumed to be
the leading jet and index 1 the subleading jet.
"""

from __future__ import annotations

import awkward as ak
import numpy as np


REQUIRED_FIELDS = (
    "jet_px",
    "jet_py",
    "jet_pz",
    "constituent_energy",
    "constituent_px",
    "constituent_py",
    "constituent_pz",
)


def _require_valid_input(data: ak.Array) -> None:
    """Check that all required fields and exactly two jets are present."""

    missing = [field for field in REQUIRED_FIELDS if field not in ak.fields(data)]
    if missing:
        raise ValueError(f"Input cache is missing required fields: {missing}")

    for field in ("jet_px", "jet_py", "jet_pz"):
        if not ak.all(ak.num(data[field], axis=1) == 2):
            raise ValueError(
                f"Field '{field}' must contain exactly two jets per event"
            )

    constituent_fields = (
        "constituent_energy",
        "constituent_px",
        "constituent_py",
        "constituent_pz",
    )
    reference_counts = ak.num(data.constituent_energy, axis=2)

    for field in constituent_fields[1:]:
        if not ak.all(ak.num(data[field], axis=2) == reference_counts):
            raise ValueError(
                "Constituent energy and momentum arrays must have matching "
                f"lengths; mismatch found in '{field}'"
            )


def _theta_between(first: ak.Array, second: ak.Array) -> ak.Array:
    """Return the energy-normalised angular distance between constituents."""

    energy_product = first.energy * second.energy
    safe_energy_product = ak.where(energy_product > 0.0, energy_product, 1.0)

    lorentz_dot = (
        energy_product
        - first.px * second.px
        - first.py * second.py
        - first.pz * second.pz
    )

    theta_squared = 2.0 * lorentz_dot / safe_energy_product
    theta_squared = np.maximum(theta_squared, 0.0)
    theta = np.sqrt(theta_squared)

    # A non-positive-energy constituent has zero energy fraction and therefore
    # contributes nothing to an energy-correlation function.
    return ak.where(energy_product > 0.0, theta, 0.0)


def _calculate_e2_e3(data: ak.Array, beta: float) -> tuple[ak.Array, ak.Array]:
    """Calculate e2 and e3 for both selected jets."""

    constituent_energy_sum = ak.sum(data.constituent_energy, axis=2)
    if not ak.all(constituent_energy_sum > 0.0):
        raise ValueError(
            "At least one selected jet has non-positive total constituent energy"
        )

    energy_fraction = (
        data.constituent_energy
        / constituent_energy_sum[:, :, np.newaxis]
    )

    constituents = ak.zip(
        {
            "z": energy_fraction,
            "energy": data.constituent_energy,
            "px": data.constituent_px,
            "py": data.constituent_py,
            "pz": data.constituent_pz,
        }
    )

    pairs = ak.combinations(
        constituents,
        2,
        axis=2,
        fields=("first", "second"),
    )
    pair_angles = _theta_between(pairs.first, pairs.second)
    e2 = ak.sum(
        pairs.first.z * pairs.second.z * pair_angles**beta,
        axis=2,
    )

    triples = ak.combinations(
        constituents,
        3,
        axis=2,
        fields=("first", "second", "third"),
    )
    theta_12 = _theta_between(triples.first, triples.second)
    theta_13 = _theta_between(triples.first, triples.third)
    theta_23 = _theta_between(triples.second, triples.third)

    e3 = ak.sum(
        triples.first.z
        * triples.second.z
        * triples.third.z
        * (theta_12 * theta_13 * theta_23) ** beta,
        axis=2,
    )

    return e2, e3


def add_hl_observables(
    data: ak.Array,
    beta: float = 0.1,
    minimum_e2: float = 1e-12,
) -> ak.Array:
    """Return only the selected high-level observables for both jets.

    Parameters
    ----------
    data
        Event records containing two selected jets and their constituents.
    beta
        Angular exponent used for e2 and e3.  The default is 0.1.
    minimum_e2
        Values of e2 at or below this threshold give NaN for C2 and D2,
        because the ratios are undefined or numerically unstable.

    Returns
    -------
    ak.Array
        Event records with exactly the fields ``e2``, ``e3``, ``jet_pt``,
        ``jet_p``, ``c2``, ``d2``, and ``jet_theta``.  Each field contains
        two values per event.
    """

    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if minimum_e2 <= 0.0:
        raise ValueError("minimum_e2 must be positive")

    _require_valid_input(data)

    jet_pt = np.sqrt(data.jet_px**2 + data.jet_py**2)
    jet_p = np.sqrt(jet_pt**2 + data.jet_pz**2)
    jet_theta = np.arctan2(jet_pt, data.jet_pz)

    e2, e3 = _calculate_e2_e3(data, beta)

    valid_ratio = (
        np.isfinite(e2)
        & np.isfinite(e3)
        & (e2 > minimum_e2)
    )
    safe_e2 = ak.where(valid_ratio, e2, 1.0)

    c2 = ak.where(valid_ratio, e3 / safe_e2**2, np.nan)
    d2 = ak.where(valid_ratio, e3 / safe_e2**3, np.nan)

    # depth_limit=1 keeps one record per event, with each field containing the
    # leading- and subleading-jet values.
    return ak.zip(
        {
            "e2": e2,
            "e3": e3,
            "jet_pt": jet_pt,
            "jet_p": jet_p,
            "c2": c2,
            "d2": d2,
            "jet_theta": jet_theta,
        },
        depth_limit=1,
    )
