import awkward as ak

sig_jets = ak.from_parquet("cache/signal_jet.parquet")
sig_particles = ak.from_parquet("cache/signal_jetParticles.parquet")

bkg_jets = ak.from_parquet("cache/background_jet.parquet")
bkg_particles = ak.from_parquet("cache/background_jetParticles.parquet")

sig_mass_record = ak.from_parquet("cache/signal_event_mass.parquet")
bkg_mass_record = ak.from_parquet("cache/background_event_mass.parquet")

sig_mass = sig_mass_record["event_invariant_mass"]
bkg_mass = bkg_mass_record["event_invariant_mass"]

print(len(sig_jets), len(sig_particles), len(sig_mass))
print(len(bkg_jets), len(bkg_particles), len(bkg_mass))

assert len(sig_jets) == len(sig_particles) == len(sig_mass)
assert len(bkg_jets) == len(bkg_particles) == len(bkg_mass)

sig_mass_mask = sig_mass > 120
bkg_mass_mask = bkg_mass > 120

# Perform mass cuts
sig_jets_cut = sig_jets[sig_mass_mask]
sig_particles_cut = sig_particles[sig_mass_mask]
bkg_jets_cut = bkg_jets[bkg_mass_mask]
bkg_particles_cut = bkg_particles[bkg_mass_mask]
# Isolate jet energies
sig_jet_energy = sig_jets_cut["Jet/Jet.energy"]
bkg_jet_energy = bkg_jets_cut["Jet/Jet.energy"]

# Mask for jets with two or more jets
sig_has_two_jets = ak.num(sig_jet_energy, axis=1) >= 2
bkg_has_two_jets = ak.num(bkg_jet_energy, axis=1) >= 2
# Apply >= 2 mask
sig_jets_cut = sig_jets_cut[sig_has_two_jets]
sig_particles_cut = sig_particles_cut[sig_has_two_jets]
sig_jet_energy = sig_jet_energy[sig_has_two_jets]
bkg_jets_cut = bkg_jets_cut[bkg_has_two_jets]
bkg_particles_cut = bkg_particles_cut[bkg_has_two_jets]
bkg_jet_energy = bkg_jet_energy[bkg_has_two_jets]

# Find indices of correct jets
sig_leading_indices = ak.argsort(sig_jet_energy,axis=1,ascending=False,)[:, :2]
bkg_leading_indices = ak.argsort(bkg_jet_energy,axis=1,ascending=False,)[:, :2]

# Apply 2jet cut to constituatent parts
sig_begin = sig_particles_cut["Jet/Jet.particles_begin"][sig_leading_indices]
sig_end = sig_particles_cut["Jet/Jet.particles_end"][sig_leading_indices]
bkg_begin = bkg_particles_cut["Jet/Jet.particles_begin"][bkg_leading_indices]
bkg_end = bkg_particles_cut["Jet/Jet.particles_end"][bkg_leading_indices]

# Validate selected indices before extraction
assert ak.all(ak.num(sig_begin, axis=1) == 2)
assert ak.all(ak.num(sig_end, axis=1) == 2)

assert ak.all(ak.num(bkg_begin, axis=1) == 2)
assert ak.all(ak.num(bkg_end, axis=1) == 2)

assert ak.all(sig_end >= sig_begin)
assert ak.all(bkg_end >= bkg_begin)

n_jets_energy = ak.num(sig_jets["Jet/Jet.energy"],axis=1,)
n_jets_ranges = ak.num(sig_particles["Jet/Jet.particles_begin"],axis=1,)
print("Jet multiplicities match:", ak.all(n_jets_energy == n_jets_ranges))
print("Begin/end shapes match:",
        ak.all(
            ak.num(sig_particles["Jet/Jet.particles_begin"], axis=1)
            ==
            ak.num(sig_particles["Jet/Jet.particles_end"], axis=1)))
n_jets_energy = ak.num(bkg_jets["Jet/Jet.energy"],axis=1,)
n_jets_ranges = ak.num(bkg_particles["Jet/Jet.particles_begin"],axis=1,)
print("Jet multiplicities match:", ak.all(n_jets_energy == n_jets_ranges))
print("Begin/end shapes match:",
        ak.all(
            ak.num(bkg_particles["Jet/Jet.particles_begin"], axis=1)
            ==
            ak.num(bkg_particles["Jet/Jet.particles_end"], axis=1)))


def extract_four_vectors_2jets(particle_data, begin, end):
    px = particle_data[
        "ReconstructedParticles/ReconstructedParticles.momentum.x"
    ]
    py = particle_data[
        "ReconstructedParticles/ReconstructedParticles.momentum.y"
    ]
    pz = particle_data[
        "ReconstructedParticles/ReconstructedParticles.momentum.z"
    ]
    energy = particle_data[
        "ReconstructedParticles/ReconstructedParticles.energy"
    ]

    selected_px = []
    selected_py = []
    selected_pz = []
    selected_energy = []

    for event in range(len(particle_data)):
        if event % 10000 == 0:
            print(f"Extracting event {event:,} / {len(particle_data):,}")
    
        event_px = []
        event_py = []
        event_pz = []
        event_energy = []

        for jet in range(2):
            particle_begin = int(begin[event][jet])
            particle_end = int(end[event][jet])

            event_px.append(
                px[event][particle_begin:particle_end]
            )
            event_py.append(
                py[event][particle_begin:particle_end]
            )
            event_pz.append(
                pz[event][particle_begin:particle_end]
            )
            event_energy.append(
                energy[event][particle_begin:particle_end]
            )

        selected_px.append(event_px)
        selected_py.append(event_py)
        selected_pz.append(event_pz)
        selected_energy.append(event_energy)

    return ak.Array({
        "px": selected_px,
        "py": selected_py,
        "pz": selected_pz,
        "energy": selected_energy,
    })


sig_four_vectors = extract_four_vectors_2jets(
    sig_particles_cut,
    sig_begin,
    sig_end,
)
bkg_four_vectors = extract_four_vectors_2jets(
    bkg_particles_cut,
    bkg_begin,
    bkg_end,
)

print(ak.type(sig_four_vectors))
print(sig_four_vectors[:1])
print(
    "Jets per event:",
    ak.to_list(ak.num(sig_four_vectors["energy"], axis=1)[:10]),
)
print(ak.type(bkg_four_vectors))
print(bkg_four_vectors[:1])
print(
    "Jets per event:",
    ak.to_list(ak.num(bkg_four_vectors["energy"], axis=1)[:10]),
)

sig_mass_cut = sig_mass[sig_mass_mask][sig_has_two_jets]
bkg_mass_cut = bkg_mass[bkg_mass_mask][bkg_has_two_jets]
sig_output = ak.with_field(
    sig_four_vectors,
    sig_jet_energy[sig_leading_indices],
    "jet_energy",
)
bkg_output = ak.with_field(
    bkg_four_vectors,
    bkg_jet_energy[bkg_leading_indices],
    "jet_energy",
)
sig_output = ak.with_field(
    sig_output,
    sig_mass_cut,
    "event_invariant_mass",
)
bkg_output = ak.with_field(
    bkg_output,
    bkg_mass_cut,
    "event_invariant_mass",
)

ak.to_parquet(
    sig_output,
    "cache/mass_cut/signal_leading_two_jet_constituents.parquet",
    compression=None,
)
ak.to_parquet(
    bkg_output,
    "cache/mass_cut/bkg_leading_two_jet_constituents.parquet",
    compression=None,
)