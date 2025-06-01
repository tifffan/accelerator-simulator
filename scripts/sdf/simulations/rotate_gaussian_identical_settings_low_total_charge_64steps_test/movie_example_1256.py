from impact import Impact
from distgen import Generator

# Make new write_beam elements and add them to the lattice.
from impact.lattice import new_write_beam

import numpy as np
import os

from bokeh.plotting import show, figure, output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh import palettes, colors
from bokeh.models.widgets import Slider

output_notebook(verbose=False, hide_banner=True)

IMPACT_IN = "./ImpactT_PR10241.in"
DISTGEN_IN = "./distgen.yaml"
os.path.exists(IMPACT_IN)

G = Generator(DISTGEN_IN)
G["n_particle"] = 2000
G.run()
P0 = G.particles
P0.plot("x", "y")

print("Generator built.")

# Make Impact object
I = Impact(IMPACT_IN, initial_particles=P0, verbose=True)

# Change some things
I.header["Nx"] = 32
I.header["Ny"] = 32
I.header["Nz"] = 32
I.header["Dt"] = 1e-13

I.total_charge = P0["charge"]
print("I.stop = ", I.stop) # Change stop location 

print("Impact object built.")

# Make a list of s values to add new write_beam elements
s_stop = 0.942 #0.942084
s_values = np.linspace(0, s_stop, 76)[1:-1] #np.linspace(0, s_stop, 80)[1:-1]

for s in s_values:
    print("Adding s value:", s)
    ele = new_write_beam(
        s=s, ref_eles=I.lattice
    )  # ref_eles will ensure that there are no naming conflicts
    I.add_ele(ele)

print("Impact object edited. Running impact...")
    
I.timeout = 1000
I.run()

print("Impact run finished. Number of particles saved:", len(I.particles))

# Print the s values at the end
print("s values added to the lattice:")
print(s_values)
