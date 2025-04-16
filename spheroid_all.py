# Script to execute both the design determination step as well as the geometry generation

import spheroid_da #TODO change this name to a more visually speaking one? spheroid_geometry I understand it generates a geometry. da I don't understand what it does
import spheroid_geometry

def run(spheroid_diameter, mini_luer, nr_of_traps, channel_negative, svg_output_file, dxf_output_file, stl_output_file, filename):
    # spheroid_da.calculate_channel_length_1D_network(spheroid_diameter, nr_of_traps)
    Q_ab = spheroid_da.generate_1D_geometry(spheroid_diameter, nr_of_traps, mini_luer, filename)
    spheroid_geometry.generate_geometry(filename, spheroid_diameter, svg_output_file, dxf_output_file, stl_output_file, mini_luer, nr_of_traps, channel_negative)

    return Q_ab

# This is where your geometry is configured:
if __name__ == '__main__':
    spheroid_diameter = 800e-6 # This needs to be smaller than the luer diameter/1.25
    mini_luer = True # Do you want mini-Luers or regular Luers
    nr_of_traps = 1 # How many traps do you want in serie
    channel_negative = True # Negative = channel cut extruded in a chip, positive = channel only for sims

    svg_output_file = 'output.svg'
    dxf_output_file = 'output.dxf'
    stl_output_file = 'output.stl'
    filename = './spheroid_trap_network.json'

    Q_ab = run(spheroid_diameter, mini_luer, nr_of_traps, channel_negative, svg_output_file, dxf_output_file, stl_output_file, filename)

    