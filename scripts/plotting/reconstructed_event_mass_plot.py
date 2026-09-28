import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import uproot

from data_processing.build_analysis_data_refactored import load_splits


MASS_CUT = 120.0
SPLIT_NAMES = ("train", "validation", "test")

# Load original ROOT-file paths from the manifest
splits = load_splits("cache/dataset_splits.json")


def get_all_files(sample):
    """
    Combine train, validation and test ROOT files for one sample.
    """
    return [
        file_path
        for split_name in SPLIT_NAMES
        for file_path in splits[sample][split_name]
    ]


def calculate_event_masses(input_files):
    """
    Calculate reconstructed event invariant mass directly
    from the original ROOT files.
    """

    masses = []

    branches = [
        "ReconstructedParticles/ReconstructedParticles.energy",
        "ReconstructedParticles/ReconstructedParticles.momentum.x",
        "ReconstructedParticles/ReconstructedParticles.momentum.y",
        "ReconstructedParticles/ReconstructedParticles.momentum.z",
    ]

    for file_number, input_file in enumerate(input_files, start=1):

        print(
            f"Reading file {file_number}/{len(input_files)}: "
            f"{input_file}"
        )

        with uproot.open(input_file) as root_file:

            tree = root_file["events"]

            data = tree.arrays(
                branches,
                library="ak",
            )

            E = data[
                "ReconstructedParticles/ReconstructedParticles.energy"
            ]
            px = data[
                "ReconstructedParticles/ReconstructedParticles.momentum.x"
            ]
            py = data[
                "ReconstructedParticles/ReconstructedParticles.momentum.y"
            ]
            pz = data[
                "ReconstructedParticles/ReconstructedParticles.momentum.z"
            ]

            # Sum reconstructed four-momentum over all particles
            E_event = ak.sum(E, axis=1)
            px_event = ak.sum(px, axis=1)
            py_event = ak.sum(py, axis=1)
            pz_event = ak.sum(pz, axis=1)

            # Event invariant mass squared
            mass_squared = (
                E_event**2
                - px_event**2
                - py_event**2
                - pz_event**2
            )

            # Protect against tiny negative numerical values
            event_mass = np.sqrt(
                np.maximum(
                    ak.to_numpy(mass_squared),
                    0.0,
                )
            )

            event_mass = event_mass[np.isfinite(event_mass)]

            masses.append(event_mass)

    return np.concatenate(masses)


# ------------------------------------------------------------------
# Original ROOT files
# ------------------------------------------------------------------

signal_files = get_all_files("signal")
background_files = get_all_files("background")

print(f"Signal ROOT files: {len(signal_files)}")
print(f"Background ROOT files: {len(background_files)}")


# ------------------------------------------------------------------
# Calculate event invariant masses
# ------------------------------------------------------------------

print("\nCalculating signal masses")
sig_mass = calculate_event_masses(signal_files)

print("\nCalculating background masses")
bkg_mass = calculate_event_masses(background_files)


# ------------------------------------------------------------------
# Apply mass selection
# ------------------------------------------------------------------

sig_mass_cut = sig_mass[sig_mass > MASS_CUT]
bkg_mass_cut = bkg_mass[bkg_mass > MASS_CUT]

print("\nEvent counts")
print(f"Signal before cut:     {len(sig_mass)}")
print(f"Signal after cut:      {len(sig_mass_cut)}")
print(f"Background before cut: {len(bkg_mass)}")
print(f"Background after cut:  {len(bkg_mass_cut)}")

print("\nSelection efficiencies")
print(
    f"Signal:     {len(sig_mass_cut) / len(sig_mass):.4f}"
)
print(
    f"Background: {len(bkg_mass_cut) / len(bkg_mass):.4f}"
)


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

fig, ax = plt.subplots(
    1,
    2,
    figsize=(14, 5),
)


# Before selection
bins_before = np.linspace(80, 130, 101)

ax[0].hist(
    sig_mass,
    bins=bins_before,
    density=True,
    alpha=0.5,
    label="Signal",
)

ax[0].hist(
    bkg_mass,
    bins=bins_before,
    density=True,
    alpha=0.5,
    label="Background",
)

ax[0].axvline(
    MASS_CUT,
    linestyle="--",
    linewidth=1.5,
    label=r"$m_{\mathrm{event}} = 120$ GeV",
)

ax[0].set_title("Before event-mass selection")
ax[0].set_xlabel(r"Event invariant mass [GeV]")
ax[0].set_ylabel("Normalised density")
ax[0].legend()


# After selection
bins_after = np.linspace(120, 130, 101)

ax[1].hist(
    sig_mass_cut,
    bins=bins_after,
    density=True,
    alpha=0.5,
    label="Signal",
)

ax[1].hist(
    bkg_mass_cut,
    bins=bins_after,
    density=True,
    alpha=0.5,
    label="Background",
)

ax[1].set_title(
    r"After $m_{\mathrm{event}} > 120$ GeV selection"
)

ax[1].set_xlabel(r"Event invariant mass [GeV]")
ax[1].set_ylabel("Normalised density")
ax[1].legend()


plt.tight_layout()

plt.savefig(
    "outputs/plots/cut_data/event_mass_before_after.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()