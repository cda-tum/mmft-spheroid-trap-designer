# Automated Calculations for the Geometry Definition of a Spheroid Trap 
# based on Quintard 2024

# Overview of the chip (top view):

#  /¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|
# |       |¯¯|  |¯¯|       |
# |   x   |  |  |  |   x   |
# |       |  |  |  |       |
# |   x---   |O-|   ---x   |
# |          |  |          |
# |   x      |  |      x   |
# |          |__|          |
# |________________________|


from z3 import *
import json
import matplotlib.pyplot as plt

eps = 1e-10

def parameter_calculation(spheroid_diameter: float, l_b: float, w_I: float, w_b: float, w_a: float, h: float) -> tuple[float, float, float]:
    '''
    Employs the z3 solver to calculate the appropriate parameters for a spheroid trap based on a given spheroid diameter to efficiently trap spheroids.
    '''
    small_chip_width = 15e-3
    l_a = Real('l_a')
    Q_ab = Real('Q_ab')

    Q_ab_max = 0.25

    K = If(w_I / h < 0.33,
            10.15,
            9.70 + 18 * (w_b / h) ** 1.1 # technically this only holds true for 0.33 < (w_b / h) < 0.8
            )

    if h/w_a <= 1:
        alpha_a = h / w_a
    else:
        alpha_a = w_a / h

    alpha_b = If(h/w_b <=1,
                h / w_b,
                w_b / h
                )

    gamma_b = w_b / w_I

    P_0a = 24 * (1 - 1.3553 * alpha_a + 1.9467 * alpha_a**2 - 1.7012 * alpha_a**3 + 0.9564 * alpha_a**4 - 0.2537 * alpha_a**5)
    P_0b = 24 * (1 - 1.3553 * alpha_b + 1.9467 * alpha_b**2 - 1.7012 * alpha_b**3 + 0.9564 * alpha_b**4 - 0.2537 * alpha_b**5)

    # l_a_max = 2 * 7005e-6 # + spheroid trap length - channel width - channel length loss due to rounding (for 15 x 15 chip)
    # radius_channel = max(spheroid_diameter * 2 - w_I/2, 80e-6) + (w_I) / 2 - math.sqrt((w_I / 2) ** 2 - (w_b / 2) ** 2) + w_I / 2 + l_b + 4/3 * spheroid_diameter
    l_b_wo_overlap = l_b - 2 * (w_I / 2 - math.sqrt((w_I / 2) ** 2 - (w_b / 2) ** 2)) # this is the length of the channel without the overlap of the trap and the channel
    radius_channel = max(spheroid_diameter * 2 - w_I/2, 80e-6) + (w_I) / 2 - math.sqrt((w_I / 2) ** 2 - (w_b / 2) ** 2) + w_I / 2 + l_b_wo_overlap + 4/3 * spheroid_diameter
    clamp_size = 1e-3
    distance_side = 1e-3

    # l_a_max = 2 * (0.5 * 15e-3 - distance_side - clamp_size + (0.5 * math.pi * radius_channel - 2 * radius_channel) - 0.5 * 4/3 * spheroid_diameter) # channel_width
    l_a_max = 2 * (0.5 * 15e-3 - distance_side - clamp_size + (0.5 * math.pi * radius_channel - radius_channel) - 0.5 * 4/3 * spheroid_diameter) # channel_width

    # l_a_max = 2 * (0.5 * 15e-3 - distance_side - clamp_size + (0.5 * math.pi * radius_channel - 2 * radius_channel) - 0.5 * 4/3 * spheroid_diameter) # channel_width


    # l_a_max = 2 * 0.5 * 11e-3 + (math.pi * radius_channel - 2 * radius_channel) - 2 * 0.5 * 4/3 * spheroid_diameter # channel_width
    print("l_a_max: ", l_a_max)
    Q_ab_exp = ((((w_b + h)**2 * (P_0b * l_b)) / (2 * w_b**3 * h**3)) + (((1 - gamma_b) * K) / (w_b**2 * h))) / (((w_a + h)**2 * (P_0a * l_a)) / (2 * w_a**3 * h**3))

    l_a_min = 2 * radius_channel # This is only required to be able to have a symmetric channel design output
    # print("Q_ab_exp: ", ((((w_b + h)**2 * (P_0b * l_b)) / (2 * w_b**3 * h**3)) + (((1 - gamma_b) * K) / (w_b**2 * h))) / (((w_a + h)**2 * (P_0a * 14e-3)) / (2 * w_a**3 * h**3)))

    s = Solver()
    # Add constraints
    # s.add(w_b <= 300e-6)
    s.add(l_a >= l_a_min)
    # s.add(l_a > 0)
    s.add(l_a < l_a_max)
    s.add(Q_ab < Q_ab_max)
    s.add(Q_ab > 0)
    s.add(Q_ab == Q_ab_exp)


    if s.check() == sat:
        model = s.model()
        print("\nSolution found:\n")
        # print(f"w_b = {model[w_b]}")
        Q_ab_value = model[Q_ab].as_decimal(50).rstrip('?')
        l_a_value = model[l_a].as_decimal(50).rstrip('?')
        
        # Convert to float and then to scientific notation
        Q_ab_sci = "{:.2f}".format(float(Q_ab_value))
        l_a_sci = "{:.2e}".format(float(l_a_value))

        print(f"Q_ab = {Q_ab_sci}")
        print(f"l_a = {l_a_sci} m")
    else:
        print("No solution found for small module 15x15\n")
        print("Checking for module size 15x30")
        l_a_max += (7.5e-3 * 2)
        # l_a_max = 2 * 15e-3 - distance_side * 2 - clamp_size * 2 + (math.pi * radius_channel - 2 * radius_channel) # TODO write this besser
        s = Solver()
        s.add(l_a >= l_a_min)
        s.add(l_a < l_a_max)
        s.add(Q_ab < Q_ab_max)
        s.add(Q_ab > 0)
        s.add(Q_ab == Q_ab_exp)
        
        if s.check() == sat:
            model = s.model()
            print("\nSolution found:\n")
            small_chip_width = 30e-3
            # print(f"w_b = {model[w_b]}")
            Q_ab_value = model[Q_ab].as_decimal(50).rstrip('?')
            l_a_value = model[l_a].as_decimal(50).rstrip('?')
            
            # Convert to float and then to scientific notation
            Q_ab_sci = "{:.2f}".format(float(Q_ab_value))
            l_a_sci = "{:.2e}".format(float(l_a_value))

            print(f"Q_ab = {Q_ab_sci}")
            print(f"l_a = {l_a_sci} m")
        else: 
            print("\n No solution found for large module 15x30\n")
            print("\n If you need help, please contact the developers\n")
            sys.exit()

    return small_chip_width, float(Q_ab_value), float(l_a_value)

def calculate_channel_length_1D_network(spheroid_diameter: float, nr_of_traps: int, mini_luer: bool):
    '''
    Calculate the required channel lengths, i.e. the distance between the nodes, for the 1D network.
    '''
    channel_width = 4/3 * spheroid_diameter
    channel_height = 4/3 * spheroid_diameter
    l_b = 80e-6
    # w_I = 4/3 * spheroid_diameter
    w_I = max(spheroid_diameter + 200e-6, 4/3 * spheroid_diameter)
    w_b = 0.5 * spheroid_diameter

    chip_width_y, Q_ab, l_a = parameter_calculation(spheroid_diameter, l_b, w_I, w_b, channel_width, channel_height)

    nodes =[] # define all nodes in m (x, y, z) 
    z = 0.0 # z coordinate of the channel

    if mini_luer == False:
        luer_radius = 1.95e-3
    else:
        luer_radius = 1.32e-3
    
    via_radius = min(1.25 * channel_width, luer_radius - eps)
    # Channel network
    # Starting point (inlet) - center left hand side

    radius_trap = w_I / 2
    add_distance_c_trap = (w_I) / 2 - math.sqrt((w_I / 2) ** 2 - (w_b / 2) ** 2) # to have a smooth transition from the trap to the outlet

    add_distance_via_inlet = via_radius - math.sqrt((via_radius) ** 2 - (channel_width/ 2) ** 2)
    # print("add_distance_via_inlet: ", add_distance_via_inlet)
    # c_trap = w_I * 1.5 - w_I/2 # should habe approx. 1.5 times the width of the inlet, we remove w_I/2 to account for the rounded end with the radius w_I/2
    # TODO discuss with Joris, added a fail save so the trap can not be negative in length for veery small spheroids, maybe change the 80e-6 to 0.0
    # This is based on experimetal values and not theoretical values:
    c_trap = max(2 * spheroid_diameter - w_I/2, 80e-6) # The length of the trap should be approx. 2 times the diameter of the spheroid, we remove w_I/2 to account for the rounded end with the radius w_I/2
    # c_short = (c_trap * 1.5 + l_b + add_distance_c_trap)
    # radius_channel = (c_trap + add_distance_c_trap + radius_trap + l_b + channel_width) / 2
    radius_channel = (c_trap + radius_trap + l_b + channel_width) / 2
    # c_short = (c_trap + add_distance_c_trap + radius_trap + l_b + channel_width)
    c_short = (c_trap + radius_trap + l_b + channel_width)
    c_long = 0.5 * (l_a - c_short - 4 * radius_channel * (0.25 * math.pi - 1))

    required_space = 2 * luer_radius + 2 * radius_channel + (1 + 2 * nr_of_traps) * c_short

    gap = 9e-3
    chip_width_x = 15e-3
    while gap < required_space:
        gap += 15e-3
        chip_width_x += 15e-3
    
    print("Chip Length set to " + str(chip_width_x) + " m")    

    chip_dimensions = [chip_width_y, chip_width_x]

    c_inlet = (gap - c_short * (1 + 2 * nr_of_traps)) / 2 - via_radius        

    # print("c_short: ", c_short)
    # print("radius_channel: ", radius_channel)
    # print("radius_trap: ", radius_trap)
    # print("c_inlet: ", c_inlet)
    # print("c_long: ", c_long)

    # Network
    nodeInlet = [3e-3 + via_radius, 7.5e-3, z]

    node0 = nodeInlet.copy()
    node0[0] += luer_radius - via_radius


    node1 = nodeInlet.copy()
    node1[0] += c_inlet - radius_channel
    node1b = nodeInlet.copy()
    node1b[0] += c_inlet
    node1b[1] += radius_channel

    node2 = node1b.copy()
    node2[1] += c_long - 2 * radius_channel
    node2b = node1b.copy()
    node2b[0] += radius_channel
    node2b[1] += c_long - radius_channel

    node3 = node2b.copy()
    node3[0] += c_short - 2 * radius_channel
    node3b = node2b.copy()
    node3b[0] += c_short - radius_channel
    node3b[1] -= radius_channel

    node4 = node3b.copy()
    node4[1] -= (c_long - radius_channel)

    node5 = node4.copy()
    node5[1] -= (c_long - radius_channel)
    node5b = node4.copy()
    node5b[0] += radius_channel
    node5b[1] -= c_long

    nodes = [nodeInlet, node0, node1, node1b, node2, node2b, node3, node3b, node4, node5, node5b]
    mirrored_nodes = []

    # print("c_trap: ", c_trap)
    # print("add_distance_c_trap: ", add_distance_c_trap)
    # print("radius_channel: ", radius_channel)
    # print("channel_width: ", channel_width)
    # distance = nodes[-1][0] - radius_channel + 0.5 * (c_trap + add_distance_c_trap + radius_trap + l_b + channel_width)
    distance = nodes[-1][0] - radius_channel + 0.5 * (c_trap + radius_trap + l_b + channel_width)

    nr_of_traps_node_gen = nr_of_traps

    for node in reversed(nodes[4:]):
        mirrored_node = mirror_nodes(node, distance)
        mirrored_nodes.append(mirrored_node)

    while nr_of_traps_node_gen > 1:
        # distance += 0.5 * (c_trap + add_distance_c_trap + radius_trap + l_b + channel_width)
        distance += 0.5 * (c_trap + radius_trap + l_b + channel_width)


        for node in nodes[8:10]:
            mirrored_node = mirror_nodes(node, distance)
            mirrored_nodes.append(mirrored_node)

        # distance += 0.5 * (c_trap + add_distance_c_trap + radius_trap + l_b + channel_width)
        distance += 0.5 * (c_trap + radius_trap + l_b + channel_width)


        mirrored_node = mirror_nodes(nodes[10], distance)
        mirrored_nodes.append(mirrored_node)

        for node in reversed(nodes[4:]):
            mirrored_node = mirror_nodes(node, distance)
            mirrored_nodes.append(mirrored_node)

        nr_of_traps_node_gen -= 1

    for node in reversed(nodes[:4]):
        mirrored_node = mirror_nodes(node, distance)
        mirrored_nodes.append(mirrored_node)

    nodes.extend(mirrored_nodes)

    # Trap Nodes
    # trap_spacing = 2 * (c_trap + add_distance_c_trap + radius_trap + l_b + channel_width)
    trap_spacing = 2 * (c_trap + radius_trap + l_b + channel_width)

    for i in range(nr_of_traps):
        node_trap1 = node4.copy()
        node_trap1[0] += c_trap + add_distance_c_trap + channel_width/2 + trap_spacing * i

        node_trap2 = node_trap1.copy()
        node_trap2[0] += radius_trap - add_distance_c_trap #+ trap_spacing * i

        nodes.append(node_trap1)
        nodes.append(node_trap2)

    # # Uncomment this to plot the nodes of the channel network
    # plot_nodes(nodes)

    return nodes, channel_width, channel_height, radius_channel, node1, w_I, w_b, chip_dimensions, Q_ab

def save_nodes_and_channels_to_json(nodes: list, channel_width: float, channel_height: float, radius_channel: float, node1: list, w_I: float, w_b: float, nr_of_traps: int, chip_dimensions: list, filename: str):
    '''
    Save the nodes and channels of the 1D network to a JSON file. 
    According to the file structure of the MMFT tools.
    '''
    # Save the network definition in a .json file format
    node_dicts = []
    for i, node in enumerate(nodes):
        node_dict = {
            "x": node[0],
            "y": node[1],
            "z": node[2]
        }
        if i == 0 or i == len(nodes) - 1 - 2 * nr_of_traps: # this accounts for the trap nodes at the end of the node list
            node_dict["ground"] = True
        node_dicts.append(node_dict)

    channel_dicts = []

    # Channel Network
    channel_dicts.append({
        "node1": 0,
        "node2": 1,
        "width": channel_width,
        "height": channel_height
    })
    channel_dicts.append({
        "node1": 1,
        "node2": 4,
        "width": channel_width,
        "height": channel_height,
        "pieces": [
            {
                "line_segment": {
                    "start": 1,
                    "end": 2
                }},{
                "arc": {
                    "right": True,
                    "start": 2,
                    "end": 3,
                    "center": [node1[0], node1[1] + radius_channel] # for now we don't have a z coordinate here
                }},{
                "line_segment": {
                    "start": 3,
                    "end": 4
                }}
        ]})
    for i in range(nr_of_traps):
        channel_dicts.append({
            "node1": 4 + i * 10,
            "node2": 8 + i * 10,
            "width": channel_width,
            "height": channel_height,
            "pieces": [
                {    
                    "arc": {
                        "right": True,
                        "start": 4 + i * 10,
                        "end": 5 + i * 10,
                        "center": [nodes[4 + i * 10][0] + radius_channel, nodes[4 + i * 10][1]]
                    }},{
                    "line_segment": {
                        "start": 5 + i * 10,
                        "end": 6 + i * 10
                    }},{
                    "arc": {
                        "right": True,
                        "start": 6 + i * 10,
                        "end": 7 + i * 10,
                        "center": [nodes[6 + i * 10][0], nodes[6 + i * 10][1] - radius_channel]
                    }},{
                    "line_segment": {
                        "start": 7 + i * 10,
                        "end": 8 + i * 10
                    }}
            ]})
        channel_dicts.append({
        "node1": 8 + i * 10,
        "node2": 13 + i * 10,
        "width": channel_width,
        "height": channel_height,
        "pieces": [
            {
                "line_segment": {
                    "start": 8 + i * 10,
                    "end": 9 + i * 10
                }},{
                "arc": {
                    "right": False,
                    "start": 9 + i * 10,
                    "end": 10 + i * 10,
                    "center": [nodes[9 + i * 10][0] + radius_channel, nodes[9 + i * 10][1]]
                }},{
                "line_segment": {
                    "start": 10 + i * 10,
                    "end": 11 + i * 10
                }},{
                "arc": {
                    "right": True,
                    "start": 11 + i * 10,
                    "end": 12 + i * 10,
                    "center": [nodes[11 + i * 10][0], nodes[11 + i * 10][1] + radius_channel]
                }},{
                "line_segment": {
                    "start": 12 + i * 10,
                    "end": 13 + i * 10
                }}
            ]})
        channel_dicts.append({
            "node1": 13 + i * 10,
            "node2": 14 + i * 10,
            "width": channel_width,
            "height": channel_height
        })
    
    closing_channel_segment_node_nr = len(nodes) - 2 * nr_of_traps - 9

    channel_dicts.append({
        "node1": closing_channel_segment_node_nr,
        "node2": closing_channel_segment_node_nr + 7,
        "width": channel_width,
        "height": channel_height,
        "pieces": [
                {
                "arc": {
                    "right": True,
                    "start": closing_channel_segment_node_nr + 1,
                    "end":  closing_channel_segment_node_nr + 2,
                    "center": [nodes[closing_channel_segment_node_nr + 1][0] + radius_channel, nodes[closing_channel_segment_node_nr + 1][1]]
                }},{
                "line_segment": {
                    "start": closing_channel_segment_node_nr + 2,
                    "end": closing_channel_segment_node_nr + 3
                }},{
                "arc": {
                    "right": True,
                    "start": closing_channel_segment_node_nr + 3,
                    "end": closing_channel_segment_node_nr + 4,
                    "center": [nodes[closing_channel_segment_node_nr + 3][0], nodes[closing_channel_segment_node_nr + 3][1] - radius_channel]
                }},{
                "line_segment": {
                    "start": closing_channel_segment_node_nr + 4,
                    "end": closing_channel_segment_node_nr + 5
                }},{
                "arc": {
                    "right": False,
                    "start": closing_channel_segment_node_nr + 5,
                    "end": closing_channel_segment_node_nr + 6,
                    "center": [nodes[closing_channel_segment_node_nr + 5][0] + radius_channel, nodes[closing_channel_segment_node_nr + 5][1]]
                }},{
                "line_segment": {
                    "start": closing_channel_segment_node_nr + 6,
                    "end": closing_channel_segment_node_nr + 7    
                }}
        ]})
    
    channel_dicts.append({
        "node1": closing_channel_segment_node_nr + 7,
        "node2": closing_channel_segment_node_nr + 8,
        "width": channel_width,
        "height": channel_height
    })
    
    # Trap Geometry
    for i in range(nr_of_traps):
        trap_node_1 = len(nodes) - (nr_of_traps - i) * 2
        trap_node_2 = trap_node_1 + 1

        channel_dicts.append({
            "node1": 8 + i * 10,
            "node2": trap_node_1,
            "width": w_I, # trap_width_inlet
            "height": channel_height
        })
        channel_dicts.append({
            "node1": trap_node_2,
            "node2": 13 + i * 10,
            "width": w_b, # trap_width_outlet
            "height": channel_height
        })

    # Create the final JSON structure
    json_data = {
        "network": {
            "nodes": node_dicts,
            "channels": channel_dicts
        },
        "chip_dimensions": {
            "width_x": chip_dimensions[0],
            "width_y": chip_dimensions[1]
        }
    }


    # Write to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print("\nNetwork definition written to " + filename + ".")

def mirror_nodes(node: list, x_axis: float)-> list:
    '''
    Mirrors a node at the defined axis along the x-axis.
    '''
    mirrored_node = node.copy()
    distance = abs(mirrored_node[0] - x_axis)
    mirrored_node[0] += 2 * distance
    return mirrored_node

def plot_nodes(nodes: list):
    '''
    Plots the 1D network, including the nodes and channels in the xy plane.
    '''
    # Extract X and Y coordinates from nodes
    x_coords = [node[0] for node in nodes]
    y_coords = [node[1] for node in nodes]

    plt.figure(figsize=(10, 8))  # Set figure size for better visibility
    plt.scatter(x_coords, y_coords, marker='.', label='Nodes')

    # Label each node with its index
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, f' {i}', color='blue', fontsize=9, ha='right', va='bottom')

    plt.title('Node Positions in XY Plane')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def generate_1D_geometry(spheroid_diameter: float, nr_of_traps: int, mini_luer: bool, filename: str):
    nodes, channel_width, channel_height, radius_channel, node1, w_I, w_b, chip_dimensions, Q_ab = calculate_channel_length_1D_network(spheroid_diameter, nr_of_traps, mini_luer)
    save_nodes_and_channels_to_json(nodes, channel_width, channel_height, radius_channel, node1, w_I, w_b, nr_of_traps, chip_dimensions, filename)

    return Q_ab

if __name__ == "__main__":
    nr_of_traps = 2
    spheroid_diameter = 400e-6
    mini_luer = False

    nodes, channel_width, channel_height, radius_channel, node1, w_I, w_b, chip_dimensions, Q_ab = calculate_channel_length_1D_network(spheroid_diameter, nr_of_traps, mini_luer)
    save_nodes_and_channels_to_json(nodes, channel_width, channel_height, radius_channel, node1, w_I, w_b, nr_of_traps, chip_dimensions, filename='./spheroid_trap_network1D.json')
