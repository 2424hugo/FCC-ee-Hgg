import awkward as ak
import numpy as np
import uproot

from scripts.data_processing.build_analysis_data_refactored import load_splits


MASS_CUT = 120.0
N_FILES = 10
SPLIT_NAMES = ("train", "validation", "test")

splits = load_splits("cache/dataset_splits.json")


def get_all_files(sample):
    """
    Combine train, validation and test ROOT-file paths for one sample.
    """
    return [
        file_path
        for split_name in SPLIT_NAMES
        for file_path in splits[sample][split_name]
    ]


def process_files(input_files):
    """
    Count events passing:
        m_event > 120 GeV
        N_jets >= 2

    Also return intermediate counts for a cut-flow.
    """

    n_total = 0
    n_mass_pass = 0
    n_twojet_pass = 0
    n_combined_pass = 0

    branches = [
        "ReconstructedParticles/ReconstructedParticles.energy",
        "ReconstructedParticles/ReconstructedParticles.momentum.x",
        "ReconstructedParticles/ReconstructedParticles.momentum.y",
        "ReconstructedParticles/ReconstructedParticles.momentum.z",
        "Jet/Jet.energy",
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

            # ------------------------------------------------------
            # Event invariant mass
            # ------------------------------------------------------

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

            E_event = ak.sum(E, axis=1)
            px_event = ak.sum(px, axis=1)
            py_event = ak.sum(py, axis=1)
            pz_event = ak.sum(pz, axis=1)

            mass_squared = (
                E_event**2
                - px_event**2
                - py_event**2
                - pz_event**2
            )

            event_mass = np.sqrt(
                np.maximum(
                    ak.to_numpy(mass_squared),
                    0.0,
                )
            )

            # ------------------------------------------------------
            # Number of reconstructed jets
            # ------------------------------------------------------

            jet_energy = data["Jet/Jet.energy"]

            n_jets = ak.to_numpy(
                ak.num(jet_energy, axis=1)
            )

            # ------------------------------------------------------
            # Selections
            # ------------------------------------------------------

            finite_mask = np.isfinite(event_mass)

            mass_mask = (
                finite_mask
                & (event_mass > MASS_CUT)
            )

            twojet_mask = (
                finite_mask
                & (n_jets >= 2)
            )

            combined_mask = (
                mass_mask
                & (n_jets >= 2)
            )

            # ------------------------------------------------------
            # Counts
            # ------------------------------------------------------

            n_total += np.sum(finite_mask)

            n_mass_pass += np.sum(mass_mask)

            n_twojet_pass += np.sum(twojet_mask)

            n_combined_pass += np.sum(combined_mask)

    return {
        "total": int(n_total),
        "mass_pass": int(n_mass_pass),
        "twojet_pass": int(n_twojet_pass),
        "combined_pass": int(n_combined_pass),
    }


def print_summary(name, result):

    total = result["total"]
    mass_pass = result["mass_pass"]
    twojet_pass = result["twojet_pass"]
    combined_pass = result["combined_pass"]

    mass_eff = mass_pass / total
    twojet_eff = twojet_pass / total
    combined_eff = combined_pass / total

    # Efficiency of >=2 jets among events already passing mass cut
    conditional_twojet_eff = (
        combined_pass / mass_pass
        if mass_pass > 0
        else 0.0
    )

    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    print(f"Total events:                  {total}")
    print(f"m_event > 120 GeV:             {mass_pass}")
    print(f"N_jets >= 2:                   {twojet_pass}")
    print(f"Both selections:               {combined_pass}")

    print("\nEfficiencies")

    print(
        f"Mass-cut efficiency:           "
        f"{mass_eff:.4f}"
    )

    print(
        f">=2-jet efficiency:             "
        f"{twojet_eff:.4f}"
    )

    print(
        f"Combined efficiency:           "
        f"{combined_eff:.4f}"
    )

    print(
        f">=2 jets after mass cut:        "
        f"{conditional_twojet_eff:.4f}"
    )


def main():

    signal_files = get_all_files("signal")[:N_FILES]
    background_files = get_all_files("background")[:N_FILES]

    print(f"Signal files used:     {len(signal_files)}")
    print(f"Background files used: {len(background_files)}")

    print("\nProcessing signal")
    signal_result = process_files(signal_files)

    print("\nProcessing background")
    background_result = process_files(background_files)

    print_summary(
        "SIGNAL",
        signal_result,
    )

    print_summary(
        "BACKGROUND",
        background_result,
    )


if __name__ == "__main__":
    main()