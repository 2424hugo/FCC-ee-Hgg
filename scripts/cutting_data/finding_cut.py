import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

# read in the data
sig = ak.from_parquet("cache/signal_event_mass.parquet")
bkg = ak.from_parquet("cache/background_event_mass.parquet")

plt.hist(
        ak.to_numpy(ak.flatten(sig, axis = None)),
        bins = 100,
        density = True,
        alpha = 0.5,
        label = 'Signal'
)
plt.hist(
        ak.to_numpy(ak.flatten(bkg, axis = None)),
        bins = 100,
        density = True,
        alpha = 0.5,
        label = 'Background'
)

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Event Mass')
plt.ylabel('Normalised Desnity')
plt.title('Event masses for background vs signal')
plt.legend()

plt.savefig(
        "outputs/plots/cut_data/bkg_vs_signal_eMass_xlog.png",
        dpi = 300
)

plt.close()


cuts = np.arange(80, 130, 0.5)

signal_eff = []
background_eff = []
significance = []

for cut in cuts:

    sig_pass = np.sum(sig["event_invariant_mass"] > cut)
    bkg_pass = np.sum(bkg["event_invariant_mass"] > cut)

    sig_eff = sig_pass / len(sig)
    bkg_eff = bkg_pass / len(bkg)

    signal_eff.append(sig_eff)
    background_eff.append(bkg_eff)

    if bkg_pass > 0:
        significance.append(sig_pass / np.sqrt(bkg_pass))
    else:
        significance.append(0)


fig, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(cuts, signal_eff,label="Signal efficiency",color="tab:blue")

ax1.plot(cuts, background_eff,label="Background efficiency",color="tab:orange")

ax1.set_xlabel("Mass cut (GeV)")
ax1.set_ylabel("Efficiency")

ax2 = ax1.twinx()

ax2.plot(cuts, significance,color="tab:red",linewidth=2,label=r"$S/\sqrt{B}$")

ax2.set_ylabel(r"$S/\sqrt{B}$")

lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]

ax1.legend(lines, labels)

plt.title("Event mass cut optimisation")
plt.tight_layout()
plt.savefig("outputs/plots/cut_data/mass_cut_optimisation.png", dpi=300)
plt.show()

best = np.argmax(significance)

print(f"Best cut = {cuts[best]:.1f} GeV")
print(f"Signal efficiency = {signal_eff[best]:.3f}")
print(f"Background efficiency = {background_eff[best]:.3f}")
print(f"S/sqrt(B) = {significance[best]:.3f}")
