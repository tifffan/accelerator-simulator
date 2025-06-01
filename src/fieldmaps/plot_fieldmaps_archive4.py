#!/usr/bin/env python3
"""
plot_fieldmaps.py

Load IMPACT-T fieldmaps (hardcoded path) and generate:
- Individual plots of Solenoid Bz, Quadrupole Bz, RF Gun Ez versus z (saved as PNG)
- A combined plot with all three curves and vertical markers at z=0 and z=0.942084 (saved as PNG)
- Discrete field values at specific z points (printed to console and saved to CSV)
"""

import numpy as np
import matplotlib.pyplot as plt
from impact import Impact, fieldmaps
import csv

# Hardcoded IMPACT-T input file path
INPUT_FILE = "/sdf/home/t/tiffan/repo/accelerator-simulator/scripts/sdf/" \
             "simulations/rotate_gaussian_vary_settings_low_total_charge/" \
             "ImpactT_PR10241.in"
# Number of z points to sample for continuous plots
NUM_POINTS = 1000
# Simulation start/stop markers
Z_START = 0.0
Z_STOP = 0.942084

def quad_fringe_field(z, z0=0.075, g=0.0508, c1=0.0004, c2=4.518219):
    """
    Compute the quadrupole fringe-field factor from IMPACT-T Equation (55).
    """
    s = (z - z0) / g
    return 1.0 / (1.0 + np.exp(c1 + c2 * s))

def main():
    # Load IMPACT-T input
    I = Impact(input_file=INPUT_FILE)

    # Extract fieldmap filenames for solenoid and gun
    sol_key = I.ele["SOL10111"]["filename"]
    gun_key = I.ele["GUNF"]["filename"]

    # Access raw fieldmap data
    fmap_sol = I.input["fieldmaps"][sol_key]
    fmap_gun = I.input["fieldmaps"][gun_key]

    # Quadrupole fringe uses formula, no raw fieldmap needed

    # Element lengths
    sol_len = I.ele["SOL10111"]["s"]
    cq_len  = I.ele["CQ10121"]["s"]  # only length needed here
    gun_len = I.ele["GUNF"]["s"]

    # Prepare continuous z-axis sampling
    # z_max = max(sol_len, cq_len, gun_len)
    z_max = Z_STOP
    zlist = np.linspace(0.0, z_max, NUM_POINTS)

    # Compute continuous profiles
    sol_profile = [
        fieldmaps.fieldmap_reconstruction_solrf(fmap_sol["field"]["Bz"], z) if z < sol_len else 0.0
        for z in zlist
    ]
    cq_profile = [
        quad_fringe_field(z) if z < cq_len else 0.0
        for z in zlist
    ]
    gun_profile = [
        fieldmaps.fieldmap_reconstruction_solrf(fmap_gun["field"]["Ez"], z) if z < gun_len else 0.0
        for z in zlist
    ]
    
    # Normalize values for each field
    def normalize(vals):
        max_abs = max(abs(v) for v in vals) or 1.0
        return [v / max_abs for v in vals]
    
    sol_profile = normalize(sol_profile)
    cq_profile = normalize(cq_profile)
    gun_profile = normalize(gun_profile)

    # Helper to add vertical markers
    def add_markers(ax):
        ax.axvline(Z_START, color='k', linestyle=':', label='_nolegend_')
        ax.axvline(Z_STOP, color='k', linestyle=':')

    # Plot and save individual and combined figures
    for profile, label, ylabel, fname in [
        (sol_profile, "Solenoid Bz", "Bz (T)", "solenoid_field.png"),
        (cq_profile, "Quadrupole Bz", "Bz (T)", "cq_field.png"),
        (gun_profile, "RF Gun Ez", "Ez (V/m)", "gun_field.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(zlist, profile, label=label)
        add_markers(ax)
        ax.set_xlabel("z (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{label} vs. z")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(fname, dpi=300)
        plt.close(fig)

    # Combined plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(zlist, sol_profile, label="Solenoid Bz")
    ax.plot(zlist, cq_profile, label="Quadrupole Bz")
    ax.plot(zlist, gun_profile, label="RF Gun Ez")
    add_markers(ax)
    ax.set_xlabel("z (m)")
    ax.set_ylabel("Field amplitude")
    ax.set_title("All Field Profiles vs. z")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("all_fields.png", dpi=300)
    plt.close(fig)

    # Generate discrete z values: start, intermediate, stop
    z_intermediate = np.linspace(0, 0.942, 76)[1:-1]
    z_values = np.hstack(([Z_START], z_intermediate, [Z_STOP]))

    # Compute discrete profiles
    sol_vals = [
        fieldmaps.fieldmap_reconstruction_solrf(fmap_sol["field"]["Bz"], z) if z < sol_len else 0.0
        for z in z_values
    ]
    cq_vals = [
        quad_fringe_field(z) if z < cq_len else 0.0
        for z in z_values
    ]
    gun_vals = [
        fieldmaps.fieldmap_reconstruction_solrf(fmap_gun["field"]["Ez"], z) if z < gun_len else 0.0
        for z in z_values
    ]

    sol_vals = normalize(sol_vals)
    cq_vals = normalize(cq_vals)
    gun_vals = normalize(gun_vals)
    
    sol_vals = [v * -1.0 for v in sol_vals]
    cq_vals = [v * 0.0 for v in cq_vals]

    # Print to console
    print("z (m), Solenoid Bz, Quadrupole Bz, RF Gun Ez")
    for z, s, q, g in zip(z_values, sol_vals, cq_vals, gun_vals):
        print(f"{z:.6f}, {s:.6e}, {q:.6e}, {g:.6e}")

    # Save to CSV
    with open("field_values.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["z (m)", "Solenoid Bz (T)", "Quadrupole Bz (T)", "RF Gun Ez (V/m)"])
        for z, s, q, g in zip(z_values, sol_vals, cq_vals, gun_vals):
            writer.writerow([f"{z:.6f}", f"{s:.6e}", f"{q:.6e}", f"{g:.6e}"])

    print("Saved CSV: field_values_const_settings.csv")
    print("Saved PNGs: solenoid_field.png, cq_field.png, gun_field.png, all_fields.png")

if __name__ == "__main__":
    main()
