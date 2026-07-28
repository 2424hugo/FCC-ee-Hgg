import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

mass_cut = 120

sig_data = ak.from_parquet("cache/signal_jet.parquet")
bkg_data = ak.from_parquet("cache/background_jet.parquet")

sig_mass_record = ak.from_parquet("cache/signal_event_mass.parquet")
bkg_mass_record = ak.from_parquet("cache/background_event_mass.parquet")

sig_mass = sig_mass_record["event_invariant_mass"]
bkg_mass = bkg_mass_record["event_invariant_mass"]

# Create masks
sig_mask = sig_mass > mass_cut
bkg_mask = bkg_mass > mass_cut

sig_mass_cut = sig_mass[sig_mask]
bkg_mass_cut = bkg_mass[bkg_mask]

sig_cut = sig_data[sig_mask]
bkg_cut = bkg_data[bkg_mask]

def compare_before_after(sig_before,bkg_before,sig_after,bkg_after,xlabel,filename,bins=100):

    sig_before = to_flat_numpy(sig_before)
    bkg_before = to_flat_numpy(bkg_before)
    sig_after = to_flat_numpy(sig_after)
    bkg_after = to_flat_numpy(bkg_after)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ax[0].hist(
            sig_before, 
            bins=bins,
            density=True,
            alpha=0.5,
            label="Signal")
    ax[0].hist(
            bkg_before,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Background")

    ax[0].set_title("Before mass cut")
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("Normalised density")
    ax[0].legend()

    ax[1].hist(
            sig_after, 
            bins=bins,
            density=True,
            alpha=0.5,
            label="Signal")
    ax[1].hist(
            bkg_after,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Background")

    ax[1].set_title("After mass cut")
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("Normalised density")
    ax[1].legend()

    plt.tight_layout()

    plt.savefig(
            f"outputs/plots/cut_data/{filename}.png",
            dpi=300)
    plt.close()


def to_flat_numpy(array):
    if isinstance(array, ak.Array):
        array = ak.flatten(array, axis=None)
        array = ak.to_numpy(array)
    array = np.asarray(array)
    return array[np.isfinite(array)]

compare_before_after(
    sig_before=sig_data['Jet/Jet.particles_end']-sig_data['Jet/Jet.particles_begin'],
    bkg_before=bkg_data['Jet/Jet.particles_end']-bkg_data['Jet/Jet.particles_begin'],
    sig_after=sig_cut['Jet/Jet.particles_end']-sig_cut['Jet/Jet.particles_begin'],
    bkg_after=bkg_cut['Jet/Jet.particles_end']-bkg_cut['Jet/Jet.particles_begin'],
    xlabel="Particle multi of jet",
    filename="jet_multiplicity_before_after",
    bins=100
)

