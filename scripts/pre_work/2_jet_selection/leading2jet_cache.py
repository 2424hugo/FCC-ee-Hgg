import os
import awkward as ak

mass_cut = 120
output_dir = "cache/mass_cut"

os.makedirs(output_dir, exist_ok=True)

# Load data
sig_data = ak.from_parquet("cache/signal_jet.parquet")
bkg_data = ak.from_parquet("cache/background_jet.parquet")

sig_mass_record = ak.from_parquet("cache/signal_event_mass.parquet")
bkg_mass_record = ak.from_parquet("cache/background_event_mass.parquet")

sig_mass = sig_mass_record["event_invariant_mass"]
bkg_mass = bkg_mass_record["event_invariant_mass"]

# Event-mass selection
sig_mask = sig_mass > mass_cut
bkg_mask = bkg_mass > mass_cut

sig_cut = sig_data[sig_mask]
bkg_cut = bkg_data[bkg_mask]

sig_mass_cut = sig_mass[sig_mask]
bkg_mass_cut = bkg_mass[bkg_mask]


def select_two_highest_energy_jets(data):
    jet_energy = data["Jet/Jet.energy"]

    # Require at least two jets
    has_two_jets = ak.num(jet_energy, axis=1) >= 2
    data = data[has_two_jets]

    # Recalculate energies after applying the event mask
    jet_energy = data["Jet/Jet.energy"]

    # Indices sorted from highest to lowest energy
    order = ak.argsort(
        jet_energy,
        axis=1,
        ascending=False,
    )

    leading_two_indices = order[:, :2]

    # Select the leading two jets from every Jet field
    selected_fields = {}

    for field in data.fields:
        if field.startswith("Jet/Jet."):
            selected_fields[field] = data[field][leading_two_indices]

    return ak.Array(selected_fields), has_two_jets


sig_leading_two, sig_has_two_jets = select_two_highest_energy_jets(sig_cut)
bkg_leading_two, bkg_has_two_jets = select_two_highest_energy_jets(bkg_cut)

# Keep event masses aligned with the selected jet events
sig_selected_mass = sig_mass_cut[sig_has_two_jets]
bkg_selected_mass = bkg_mass_cut[bkg_has_two_jets]

# Optionally include event invariant mass in the same parquet record
sig_output = ak.with_field(
    sig_leading_two,
    sig_selected_mass,
    "event_invariant_mass",
)

bkg_output = ak.with_field(
    bkg_leading_two,
    bkg_selected_mass,
    "event_invariant_mass",
)

# Save parquet files
ak.to_parquet(
    sig_output,
    f"{output_dir}/signal_leading_two_jets_mass_gt_{mass_cut}.parquet",
    compression=None,
)

ak.to_parquet(
    bkg_output,
    f"{output_dir}/background_leading_two_jets_mass_gt_{mass_cut}.parquet",
    compression=None,
)

print(f"Signal events saved:     {len(sig_output)}")
print(f"Background events saved: {len(bkg_output)}")
print(sig_output.type)