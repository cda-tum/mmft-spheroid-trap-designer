# Script for the generation of an STL file for a spheroid trap 
# The luer connection is based on the pump connections of the Organ-on-chip generator

import numpy as np
from stl import mesh
import re

import os
import ezdxf
from collections import Counter, defaultdict
import math
import json

import matplotlib.pyplot as plt


eps = 1e-10

def read_in_network_file(filename: str) -> tuple[list, list, list, list, float]:
    '''
    Read in the network file and return the nodes, grounds, channels, arcs and the height of the channel network.
    '''
    nodes = []
    grounds = []
    channels = []
    arcs = []

    height = 0.0

    try:
        with open(filename) as f:
            json_data = json.load(f)
            for node in json_data['network']['nodes']: # Their order determines their ID
                nodes.append([node['x'], node['y'], node['z']])
                # nodes.append([node['x'], node['y']])
                if 'ground' in node: # i.e. if the node is defined as a ground
                    grounds.append(len(nodes)-1) # append the nodeId
            for channel in json_data['network']['channels']: # TODO this depends on the input (in ooc there are vias and rounded vias, here its line segments and arcs)
                # this is altered for the spheroid case because the line segments are not defined in realtion to the start node
                height = channel['height'] # This is only correct if all channels have the same height
                if 'pieces' in channel:
                    num_pieces = len(channel['pieces'])
                    # current_start_node = channel['node1']
                    for i in range(num_pieces):
                        piece = channel['pieces'][i]
                        if 'line_segment' in piece:
                            channels.append([piece['line_segment']['start'], piece['line_segment']['end'], channel['width'], channel['height']])
                        elif 'arc' in piece:
                            center = piece['arc']['center']
                            direction = piece['arc']['right']
                            arcs.append([piece['arc']['start'], center, piece['arc']['end'], channel['width'], channel['height'], direction])
                else:
                    channels.append([channel['node1'], channel['node2'], channel['width'], channel['height']])
            
            chip_length = json_data['chip_dimensions']['width_x']
            chip_width = json_data['chip_dimensions']['width_y']

    except FileNotFoundError:
        print(f"File {filename} not found.")

    return nodes, grounds, channels, arcs, height, chip_width, chip_length

def define_channels_per_node(nodes: list, channels: list) -> dict:
    '''
    Define the channels that are connected to each node in the channel network.
    '''
    channels_per_node = defaultdict(list)

    for node in range(len(nodes)):
        for channel in channels:
            if channel[0] == node or channel[1] == node:
                channels_per_node[node].append(channel)

    return channels_per_node

def define_quads_at_nodes(nodes: list, channels_per_node: dict) -> dict:
    '''
    Extrude the nodes to quads based on the channels connected to them, each quad is associated with a width and a length.
    '''
    # i.e.  ---3-----2
    #          |  x  | l
    #       ---0-----1
    #          |  w  |
    quads = {} # contains node ID and width and height of the quad surrounding that node

    for node in channels_per_node: # this doesn't yet include angled channels!
        quad_widths = []
        quad_lengths = []
        for channel in channels_per_node[node]:
            width = channel[2]
            # loop through all channels here 
            if math.isclose(nodes[channel[0]][1], nodes[channel[1]][1], rel_tol=eps): # horizontal channel
                # length = abs(nodes[channel[0]][1] - nodes[channel[1]][1])
                # quad_length = width
                quad_lengths.append(width)

            elif math.isclose(nodes[channel[0]][0], nodes[channel[1]][0], rel_tol=eps): # vertical channel
                # length = abs(nodes[channel[0]][0] - nodes[channel[1]][0])
                # quad_width = width
                quad_widths.append(width)
            else:
                print("Error: channel is not horizontal or vertical")
                print("channel:", channel)

        quads[node] = [quad_widths, quad_lengths]

    return quads

def define_vertices_and_quad_list(nodes: list, quads: dict, channel_height: float) -> tuple[list, list]:
    '''
    For each quad that was defined around a node the actual coordinates (vertices) are defined. This is done first for the bottom layer and then extruded to a 3D coordinate cloud.
    The vertices around a node are defined as follows:
    e.g. ---3-----2
            |  x  | l
         ---0-----1
            |  w  |
    '''
    quad_faces_xy = []
    vertices = []
    quad_list = []
    # z = nodes[0][2] # Alternatively the z-value of the first node can be used as a starting point for the 3D extrusion 
    z = 0.0
    # height = channel_height
    height = 0.0

    for j in range(2):    
        for i, node in enumerate(quads):
            if len(quads[node][0]) > 0 and len(quads[node][1]) > 0: # if the quad has a width and length
                if len(quads[node][0]) > 1:
                    # print("Error: Multiple widths at node", node)
                    width1 = quads[node][0][0]
                    width2 = quads[node][0][1]
                elif len(quads[node][0]) == 1:
                    width1 = width2 = quads[node][0][0]
                if len(quads[node][1]) > 1:
                    # print("Error: Multiple lengths at node", node)
                    length1 = quads[node][1][0]
                    length2 = quads[node][1][1]
                elif len(quads[node][1]) == 1:
                    length1 = length2 = quads[node][1][0]
                vertices.append([nodes[node][0] - width1/2, nodes[node][1] - length2/2, z + height])
                vertices.append([nodes[node][0] + width1/2, nodes[node][1] - length1/2, z + height])
                vertices.append([nodes[node][0] + width2/2, nodes[node][1] + length1/2, z + height])
                vertices.append([nodes[node][0] - width2/2, nodes[node][1] + length2/2, z + height])
                # quad_faces_xy.append([[nodes[node][0] - quads[node][0]/2, nodes[node][1] - quads[node][1]/2, nodes[node][2] + height], 
                #                      [nodes[node][0] - quads[node][0]/2, nodes[node][1] + quads[node][1]/2, nodes[node][2] + height],
                #                      [nodes[node][0] + quads[node][0]/2, nodes[node][1] + quads[node][1]/2, nodes[node][2] + height],
                #                      [nodes[node][0] + quads[node][0]/2, nodes[node][1] - quads[node][1]/2, nodes[node][2] + height]]) # vertices 0, 1, 2, 3
                quad_faces_xy.append([vertices[len(vertices) - 4], vertices[len(vertices) - 3], vertices[len(vertices) - 2], vertices[len(vertices) - 1]])
                # quad_list.append([i + 0, i + 1, i + 2, i + 3]) # ist geordnet nach nodes weil die quads nach nodes geordnet sind
                quad_list.append([len(vertices) - 4, len(vertices) - 3, len(vertices) - 2, len(vertices) - 1])
            # elif quads[node][0] == 0 and quads[node][1] > 0: # quad has no width!
            # elif quads[node][1][0] > 0 and quads[node][0][1] > 0: # quad has lengths!
            elif len(quads[node][0]) == 0: # quad has no width!
                if len(quads[node][1]) > 1:
                    # print("Error: Multiple lengths at node", node)
                    length1 = quads[node][1][0]
                    length2 = quads[node][1][1]
                elif len(quads[node][1]) == 1:
                    length1 = length2 = quads[node][1][0]
                vertices.append([nodes[node][0], nodes[node][1] - length2/2, z + height])
                vertices.append([nodes[node][0], nodes[node][1] + length1/2, z + height])
                # quad_list.append([i + 0, i + 1])
                quad_list.append([len(vertices) - 2, len(vertices) - 2, len(vertices) - 1, len(vertices) - 1])
            # elif quads[node][0] > 0 and quads[node][1] == 0: # quad has no length!
            # elif quads[node][0][0] > 0 and quads[node][0][1] > 0: # quad has widths!
            elif len(quads[node][1]) == 0: # quad has no length!
                if len(quads[node][0]) > 1:
                    # print("Error: Multiple widths at node", node)
                    width1 = quads[node][0][0]
                    width2 = quads[node][0][1]
                elif len(quads[node][0]) == 1:
                    width1 = width2 = quads[node][0][0]
                vertices.append([nodes[node][0] - width1/2, nodes[node][1], z + height])
                vertices.append([nodes[node][0] + width2/2, nodes[node][1], z + height])
                # quad_list.append([i + 0, i + 1])
                quad_list.append([len(vertices) - 2, len(vertices) - 1, len(vertices) - 1, len(vertices) - 2])
        # height = -height # This ensures that the 3D channel extrusion is equal in - z and + z direction, with the 1D definition as a center of the channel 
        height = channel_height

    return vertices, quad_list

def define_channel_faces_xy(nodes: list, channels: list, quad_list: list) -> list:
    '''
    Define the faces of the channels in the xy plane.
    '''
    channel_faces_xy = []

    # trap_channels_out = []
    # trap_channels_in = []
    # for i in range(nr_of_traps):
    #     trap_channels_out.append(channels[-(1 + i * 2)]) # TODO double check

    # for channel in channels[:-(1 + 2 * nr_of_traps)] + trap_channels_out:
    for channel in channels:
            quad_1 = quad_list[channel[0]]
            if len(quad_1) == 2:
                quad_1 = [quad_list[channel[0]][0],quad_list[channel[0]][0],quad_list[channel[0]][1],quad_list[channel[0]][1]]

            quad_2 = quad_list[channel[1]]
            if len(quad_2) == 2:
                quad_2 = [quad_list[channel[0]][0],quad_list[channel[0]][0],quad_list[channel[0]][1],quad_list[channel[0]][1]]

            if math.isclose(nodes[channel[0]][1], nodes[channel[1]][1], rel_tol=eps): # horizontal channel
                if nodes[channel[0]][0] < nodes[channel[1]][0]: # channel goes from left to right
                    channel_faces_xy.append([quad_1[1], quad_2[0], quad_2[3], quad_1[2]])
                elif nodes[channel[0]][0] > nodes[channel[1]][0]: # channel goes from right to left
                    channel_faces_xy.append([quad_2[1], quad_1[0], quad_1[3], quad_2[2]])
            elif math.isclose(nodes[channel[0]][0], nodes[channel[1]][0], rel_tol=eps): # vertical channel
                if nodes[channel[0]][1] > nodes[channel[1]][1]: # channel goes from top to bottom
                    channel_faces_xy.append([quad_2[3], quad_2[2], quad_1[1], quad_1[0]])
                elif nodes[channel[0]][1] < nodes[channel[1]][1]: # channel goes from bottom to top
                    channel_faces_xy.append([quad_1[3], quad_1[2], quad_2[1], quad_2[0]])

    return channel_faces_xy

def define_quad_faces_xy(quad_list:list) -> list:
    '''
    The quad faces are defined by the vertices of the quads in the xy plane. This is analogous to the channel face definition.
    '''
    quad_faces = []
    for quad in quad_list:
        if len(quad) == 4:
            quad_faces.append([quad[0], quad[1], quad[2], quad[3]])
    return quad_faces


def define_faces_side(faces_xy_bottom: list, faces_xy_top: list) -> list:
    '''
    The side faces are defined by the vertices of the bottom and top faces of the channels.
    '''
    faces_side = []

    if len(faces_xy_bottom) != len(faces_xy_top):
        print("Error! Number of faces on bottom and top are not equal")
        
    for i, face in enumerate(faces_xy_bottom):
        vertice_0 = faces_xy_bottom[i][0]
        vertice_1 = faces_xy_bottom[i][1]
        vertice_2 = faces_xy_bottom[i][2]
        vertice_3 = faces_xy_bottom[i][3]

        vertice_0_top = faces_xy_top[i][0]
        vertice_1_top = faces_xy_top[i][1]
        vertice_2_top = faces_xy_top[i][2]
        vertice_3_top = faces_xy_top[i][3]

        # faces_side.extend([[vertice_0, vertice_1, vertice_1_top, vertice_0_top]]) # TODO this needs to be changed for the arcs!!!
        # faces_side.extend([[vertice_1, vertice_2, vertice_2_top, vertice_1_top]])
        # faces_side.extend([[vertice_2, vertice_3, vertice_3_top, vertice_2_top]])
        # faces_side.extend([[vertice_3, vertice_0, vertice_0_top, vertice_3_top]])

        faces_side.extend([[vertice_0, vertice_1, vertice_1_top, vertice_0_top]]) # TODO this needs to be changed for the arcs!!!
        faces_side.extend([[vertice_1, vertice_2, vertice_2_top, vertice_1_top]])
        faces_side.extend([[vertice_2, vertice_3, vertice_3_top, vertice_2_top]])
        faces_side.extend([[vertice_3, vertice_0, vertice_0_top, vertice_3_top]])

        # faces_side.extend([[vertice_0, vertice_1, vertice_1_top, vertice_0_top]]) # changed all vertices in anti clockwise direction
        # faces_side.extend([[vertice_2, vertice_1, vertice_1_top, vertice_2_top]])
        # faces_side.extend([[vertice_3, vertice_2, vertice_2_top, vertice_3_top]])
        # faces_side.extend([[vertice_3, vertice_0, vertice_0_top, vertice_3_top]])

    return faces_side

def discretize_arc(start: list, center: list, end: list, direction: bool, num_segments: int, radius: float, height: float) -> list: # TODO merge the discretize arc functions
    """
    Discretize an arc into a series of points. This is used for the arcs in the channel network definition.
    """
    # center, radius, direction = find_circle_center(start, midpoint, end)
    if direction:
        offset = 1
    else:
        offset = -1

    # Define the arc angles
    vec_start = np.array(start, dtype=np.float64) - np.array(center, dtype=np.float64)
    vec_end = np.array(end, dtype=np.float64) - np.array(center, dtype=np.float64)

    if radius == 0:
        # radius = abs(start[1] - center[1] + start[0] - center[0] + start[2] - center[2])
        radius = abs(start[1] - center[1] + start[0] - center[0])
    else:
        radius = radius

    start_angle = np.arctan2(vec_start[1], vec_start[0]) * offset
    end_angle = np.arctan2(vec_end[1], vec_end[0])

    # center[2] = -height/2 # required for the 3D extrusion to be equal in +z and -z direction
    center[2] = 0.0
    arc_points = []
    for i in range(num_segments + 1):
        theta = start_angle + (end_angle - start_angle) * i / num_segments
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta) 
        arc_points.append((x, y, center[2]))

    return arc_points

def discretize_arc_2(start: list, center: list, end: list, direction: float, num_segments: int, radius: float, height: float) -> list:
    """
    Discretize an arc into a series of points along the xy plane. 
    This is used for arcs that define the chip geometry, the channel geometry is defined by the discretize_arc function.
    """
    # Define the arc angles
    vec_start = np.array(start, dtype=np.float64) - np.array(center, dtype=np.float64)
    vec_end = np.array(end, dtype=np.float64) - np.array(center, dtype=np.float64)

    if radius == 0:
        radius = abs(start[1] - center[1] + start[0] - center[0])
    else:
        radius = radius

    start_angle = np.arctan2(vec_start[1], vec_start[0]) + direction
    end_angle = np.arctan2(vec_end[1], vec_end[0])

    if height != 0:
        # center[2] = -height/2 # required for the 3D extrusion to be equal in +z and -z direction only for the channel arcs! 
        center[2] = 0.0

    arc_points = []
    for i in range(num_segments + 1):
        theta = start_angle + (end_angle - start_angle) * i / num_segments
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        arc_points.append((x, y, center[2]))

    return arc_points


def create_arc_triangles_side(arc_points: list, height: float, inner_arc: bool) -> list: # TODO define direction as well ie z, y or x
    """
    Extrudes the 1D arc in z-direction. For this, triangles for the channel sides are created that are defined by the arc points and height. 
    """
    triangles = []
 
    for i in range(len(arc_points) - 1):
        p1, p2 = np.array(arc_points[i], dtype=np.float64), np.array(arc_points[i + 1], dtype=np.float64)
        segment_vec = p2 - p1
        segment_length = np.linalg.norm(segment_vec)

        # Skip zero-length segments
        if segment_length == 0:
            continue

        p1_top = p1.copy()
        p1_top[2] += height
        p2_top = p2.copy()
        p2_top[2] += height

        if inner_arc == False:
            triangles.append([tuple(p1), tuple(p2), tuple(p1_top)]) # here the outer arcs are not defined correctly
            triangles.append([tuple(p2_top), tuple(p1_top), tuple(p2)])
        else:
            triangles.append([tuple(p2), tuple(p1), tuple(p1_top)]) # here the inner arcs are not defined correctly
            triangles.append([tuple(p2), tuple(p1_top), tuple(p2_top)])

        # triangles.append([tuple(p1), tuple(p2), tuple(p2_top)]) # here the inner arcs are not defined correctly
        # triangles.append([tuple(p1), tuple(p2_top), tuple(p1_top)])

    return triangles

def create_arc_triangles_side_angled(arc_points: list, arc_points_top: list, height: float) -> list: 
    """
    Extrudes the 1D arc in z-direction. For this, triangles for the channel sides are created that are defined by the arc points and height. Here an angle is included, i.e., the arcs are not identical in the xy plane. 
    The angle is required for the luer connection.
    """
    triangles = []
 
    for i in range(len(arc_points) - 1):
        p1, p2 = np.array(arc_points[i], dtype=np.float64), np.array(arc_points[i + 1], dtype=np.float64)
        segment_vec = p2 - p1
        segment_length = np.linalg.norm(segment_vec)

        # Skip zero-length segments
        if segment_length == 0:
            continue

        p1_top, p2_top = np.array(arc_points_top[i], dtype=np.float64), np.array(arc_points_top[i + 1], dtype=np.float64)

        p1_top[2] += height
        p2_top[2] += height

        triangles.append([tuple(p1), tuple(p2), tuple(p1_top)])
        triangles.append([tuple(p2_top), tuple(p1_top), tuple(p2)])

    return triangles

def create_arc_triangles_xy(arc_points: list, arc_points2: list, height: float, channel_negative: bool) -> list:
    '''
    Define the top and bottom of the arcs as triangles.
    '''
    triangles = []
    
    for i in range(len(arc_points) - 1):
        p1A, p2A = np.array(arc_points[i], dtype=np.float64), np.array(arc_points[i + 1], dtype=np.float64)
        p1B, p2B = np.array(arc_points2[i], dtype=np.float64), np.array(arc_points2[i + 1], dtype=np.float64)

        segment_vec = p2A - p1A
        segment_length = np.linalg.norm(segment_vec)

        # Skip zero-length segments
        if segment_length == 0:
            continue

        if channel_negative == False:
            triangles.append([tuple(p1B), tuple(p1A), tuple(p2A)])
            triangles.append([tuple(p2B), tuple(p1B), tuple(p2A)])

        # Analogous for the top/bottom triangles +z
        p1A_top = p1A.copy()
        p1A_top[2] += height
        p2A_top = p2A.copy()
        p2A_top[2] += height
        p1B_top = p1B.copy()
        p1B_top[2] += height
        p2B_top = p2B.copy()
        p2B_top[2] += height

        triangles.append([tuple(p1B_top), tuple(p1A_top), tuple(p2A_top)])
        triangles.append([tuple(p2B_top), tuple(p1B_top), tuple(p2A_top)])

    return triangles

def define_arcs(nodes: list, quad_list: list, vertices: list, arcs: list, numSegments: int, height: float, nr_of_traps: int, channel_negative: bool) -> tuple[list, list]:
    """
    Define the arc geometry using triangles by employing the discretize_arc, create_arc_triangles_side and create_arc_triangles_xy functions.
    Additionally the corner/center points of the arcs are defined for the chip definition.
    """
    arc_triangles = []
    top_arc_triangles = []
    corner_points_arc = []
    # height = arcs[0][4] # TODO adapt this for channels that have different heights

    counter = 0

    inner_arc = True # True if it is 1 false if it is 2

    for arc in arcs:
        arc[1] = arc[1] + [nodes[arc[0]][2]] # add the z-value to the center TODO add this to the json file or repeat for each arc

        quad_1 = quad_list[arc[0]]
        quad_2 = quad_list[arc[2]]

        if nodes[arc[0]][0] < nodes[arc[2]][0]: # arc goes in + x direction 
            if nodes[arc[0]][1] < nodes[arc[2]][1]: # arc goes in + y direction
                if arc[1][1] == nodes[arc[0]][1]: # arc center is in +x direction
                    end1 = quad_1[3] # 2. try
                    end2 = quad_1[2]
                    start1 = quad_2[3]
                    start2 = quad_2[0]
                    inner_arc = False
                elif arc[1][0] == nodes[arc[0]][0]: # arc center is in +y direction
                    start1 = quad_1[1]
                    start2 = quad_1[2]
                    end1 = quad_2[1]
                    end2 = quad_2[0]
                    inner_arc = False
                else:
                    print("Error: arc1a", arc) 
            elif nodes[arc[0]][1] > nodes[arc[2]][1]: # arc goes in - y direction
                if arc[1][1] == nodes[arc[0]][1]:
                    start1 = quad_1[0]
                    start2 = quad_1[1]
                    end1 = quad_2[0]
                    end2 = quad_2[3]
                elif arc[1][0] == nodes[arc[0]][0]: # arc center is in -y direction
                    end1 = quad_1[2] # 2. try
                    end2 = quad_1[1]
                    start1 = quad_2[2]
                    start2 = quad_2[3]
                else:
                    print("Error: arc1b", arc) 
            else:
                print("Error: arc2", arc) 
        elif nodes[arc[0]][0] > nodes[arc[2]][0]: # arc goes in -x direction
            if nodes[arc[0]][1] < nodes[arc[2]][1]: # arc goes in + y direction
                if arc[1][1] == nodes[arc[0]][1]: # arc center is in -x direction
                    end1 = quad_1[3] # 2. try
                    end2 = quad_1[2]
                    start1 = quad_2[1]
                    start2 = quad_2[2]
                elif arc[1][0] == nodes[arc[0]][0]: # arc center is in +y direction
                    start1 = quad_1[0]
                    start2 = quad_1[3]
                    end1 = quad_2[0]
                    end2 = quad_2[1]
                else:
                    print("Error: arc1", arc) 
            elif nodes[arc[0]][1] > nodes[arc[2]][1]: # arc goes in - y direction
                if arc[1][1] == nodes[arc[0]][1]: # arc center is in -x direction
                    end1 = quad_1[1] # 2. try
                    end2 = quad_1[0]
                    start1 = quad_2[1]
                    start2 = quad_2[2]
                elif arc[1][0] == nodes[arc[0]][0]: # arc center is in -y direction
                    end1 = quad_1[3] # 2. try
                    end2 = quad_1[0]
                    start1 = quad_2[3]
                    start2 = quad_2[2]
                else:
                    print("Error: arc1", arc)    
            else:
                print("Error: arc2", arc) 
        else:
            print("Error: arc3", arc)

        start1 = vertices[start1].copy()
        # start1[2] -= height/2
        start2 = vertices[start2].copy()
        # start2[2] -= height/2
        end1 = vertices[end1].copy()
        # end1[2] -= height/2
        end2 = vertices[end2].copy()
        # end2[2] -= height/2
            
        # arc_points_1 = discretize_arc(vertices[start1], arc[1], vertices[end1], 0, numSegments, 0)
        # arc_points_2 = discretize_arc(vertices[start2], arc[1], vertices[end2], 0, numSegments, 0)
        direction = arc[5]
        center_point = arc[1]
        center_points = []
        for i in range(numSegments):
            center_points.extend(center_point)
        arc_points_1 = discretize_arc(start1, arc[1], end1, direction, numSegments, 0, height) # the height here is only required if the 3D extrusion is equal in +z and -z direction, rather than starting at the bottom
        arc_points_2 = discretize_arc(start2, arc[1], end2, direction, numSegments, 0, height) # the height here is only required if the 3D extrusion is equal in +z and -z direction, rather than starting at the bottom
        arc_triangles_xy_top = create_arc_triangles_xy(arc_points_2, arc_points_1, height, True)
        top_arc_triangles.extend(arc_triangles_xy_top)
        if channel_negative == True: 
             # switched arc_points_1 and arc_points_2 to dirct all triangles the same way
            direction = change_arc_direction(counter, nr_of_traps, direction)
            bottom_layer_triangles, corner_point = create_arc_triangles_xy_2(arc_points_1, arc_points_2, 0.0, direction, 0) # the outward bottom layer
            top_arc_triangles.extend(bottom_layer_triangles)
            corner_points_arc.append(corner_point)
            bottom_layer_triangles, corner_point = create_arc_triangles_xy_2(arc_points_2, center_points, 0.0, not direction, 0) # the inward bottom layer
            top_arc_triangles.extend(bottom_layer_triangles)
            corner_points_arc.append(corner_point)
        else:
            arc_triangles_xy_bottom = create_arc_triangles_xy(arc_points_2, arc_points_1, 0.0, True) # switched arc_points_1 and arc_points_2 to dirct all triangles the same way
            arc_triangles.extend(arc_triangles_xy_bottom)
        arc_triangles_side_1 = create_arc_triangles_side(arc_points_1, height, not inner_arc)
        arc_triangles_side_2 = create_arc_triangles_side(arc_points_2, height, inner_arc)
        arc_triangles_side = arc_triangles_side_1 + arc_triangles_side_2
        
        arc_triangles.extend(arc_triangles_side)

        counter += 1
    return arc_triangles, top_arc_triangles, corner_points_arc

def change_arc_direction(counter: int, nr_of_traps: int, direction: bool) -> bool:
    '''
    Flips the direction of the arc based on requirements of the specific geometry.
    '''
    trap_indices = [2, 6] + [6 + i * 4 for i in range(1, nr_of_traps)]

    if counter in trap_indices:
        direction = not direction

    return direction

def trap_geometry(nodes: list, quad_list: list, vertices: list, channels: list, channel_negative: bool, num_segments: int, nr_of_traps: int) -> tuple[list, list, list]:
    '''
    Definition of the 3D trap, this is based on the already existing channels, here they are connected using arcs.
    The corner points of the trap are defined for the chip geometry.
    '''
    trap_triangles = []
    trap_triangles_bottom = []
    corner_points_trap = []
    trap_start_and_end_nodes = []
    # the rounded edges of the trap are between node 20 and 21
    radius = channels[-2][2] / 2
    center_width = channels[-1][2]
    height = channels[-1][3]

    for i in range(nr_of_traps):
        trap_node_1 = len(nodes) - (i + 1) * 2
        trap_node_2 = trap_node_1 + 1

        left_node = len(nodes) - 14 - nr_of_traps * 2 - i * 10 # because we iterate from the last trap to the first

        # These are adapted to reuse the same vertices as before!
        vertice_0 = nodes[trap_node_1]
        vertice_1 = vertices[quad_list[trap_node_2][0]]
        vertice_2 = vertices[quad_list[trap_node_2][3]]

        vertice_0_top = vertice_0.copy()
        vertice_0_top[2] += height
        vertice_1_top = vertice_1.copy()
        vertice_1_top[2] += height
        vertice_2_top = vertice_2.copy()
        vertice_2_top[2] += height

        center_points = []
        for i in range(num_segments+1):
            center_points.append([vertice_0[0], vertice_0[1], vertice_0[2]]) 

        start_left = vertices[quad_list[trap_node_1][3]]
        start_left_top = start_left.copy()
        start_left_top[2] += height
        start_right = vertices[quad_list[trap_node_1][0]]
        start_right_top = start_right.copy()
        start_right_top[2] += height

        if channel_negative == False:
            # trap_triangles.append([vertice_0, vertice_2, vertice_1])
            trap_triangles_bottom.append([vertice_2, vertice_1, vertice_0])
        
            trap_triangles_bottom.append([vertices[quad_list[left_node][1]], vertice_0, start_right])
            trap_triangles_bottom.append([start_left, vertice_0, vertices[quad_list[left_node][1]]])
            
            trap_triangles_bottom.append([start_left, vertices[quad_list[left_node][1]], vertices[quad_list[left_node][2]]])

        # trap_triangles.append([vertice_0_top, vertice_2_top, vertice_1_top])
        trap_triangles.append([vertice_2_top, vertice_1_top, vertice_0_top])

        vertices_left_node_top = vertices[quad_list[left_node][1]].copy() # TODO rename!!!!
        vertices_left_node_top[2] += height
        vertices_left_node_top2 = vertices[quad_list[left_node][2]].copy()
        vertices_left_node_top2[2] += height

        trap_triangles.append([vertices_left_node_top, vertice_0_top, start_right_top])
        trap_triangles.append([start_left_top, vertice_0_top, vertices_left_node_top])
        
        trap_triangles.append([start_left_top, vertices_left_node_top, vertices_left_node_top2])

        # arc_points_left = discretize_arc(start_left, vertice_0, vertice_1, False, num_segments, 0, height)
        arc_points_left = discretize_arc(vertice_2, vertice_0, start_right, False, num_segments, radius, height)
        arc_points_right = discretize_arc(start_right, vertice_0, vertice_2, False, num_segments, 0, height)
        # arc_points_right = discretize_arc(vertice_1, vertice_0, start_left, False, num_segments, radius, height)

        trap_start_and_end_nodes.append([start_left, vertice_2, start_right, vertice_1]) # Start, end left, start, end right
        arc_triangles_xy_left = create_arc_triangles_xy(arc_points_left, center_points, height, True)
        trap_triangles_bottom += create_arc_triangles_xy(arc_points_left, center_points, 0.0, True)
        # arc_triangles_xy_left = create_arc_triangles_xy(center_points,arc_points_left,  height, channel_negative)
        arc_triangles_xy_right = create_arc_triangles_xy(arc_points_right, center_points, height, True)
        trap_triangles_bottom += create_arc_triangles_xy(arc_points_right, center_points, 0.0, True)
        arc_triangles_side_left = create_arc_triangles_side(arc_points_left, height, False)
        arc_triangles_side_right = create_arc_triangles_side(arc_points_right, height, False)

        trap_triangles.extend(arc_triangles_xy_left + arc_triangles_xy_right + arc_triangles_side_left + arc_triangles_side_right)

        if channel_negative:
            bottom_triangles_left, corner_points_left = create_arc_triangles_xy_2(arc_points_left, center_points, 0.0, False, 0)
            bottom_triangles_right, corner_points_right = create_arc_triangles_xy_2(arc_points_right, center_points, 0.0, True, 0)
            trap_triangles.extend(bottom_triangles_left + bottom_triangles_right)
            corner_points_trap.append(corner_points_left)
            corner_points_trap.append(corner_points_right)

    return trap_triangles, trap_triangles_bottom, corner_points_trap, trap_start_and_end_nodes

def triangulation(faces: list, vertices: list, xy_orientation: bool) -> list: # TODO the xy_orientation is not used
    '''
    Triangulates faces. This means the rectangular face is cut into two triangles.
    '''
    triangles = []
    for face in faces:
        vertice_0 = vertices[face[0]]
        vertice_1 = vertices[face[1]]
        vertice_2 = vertices[face[2]]
        vertice_3 = vertices[face[3]]

        # if xy_orientation:
        #     triangles.append([vertice_0, vertice_1, vertice_2])
        #     triangles.append([vertice_0, vertice_2, vertice_3])
        # else:
        #     triangles.append([vertice_0, vertice_1, vertice_3])
        #     triangles.append([vertice_0, vertice_2, vertice_3])

        triangles.append([vertice_1, vertice_0, vertice_2])
        triangles.append([vertice_2, vertice_0, vertice_3])
        
    return triangles

def remove_shared_faces(all_faces: list) -> list: 
    '''
    Creates a dictionary to count faces. The original list is stored but all faces that occur more than once are removed without changing the original order.
    This is necessary to make sure the channel network is self-contained without any internal faces cutting through the geometry.
    '''
    face_count = {}
    for face in all_faces:
        sorted_face = tuple(sorted(face))
        if sorted_face in face_count:
            face_count[sorted_face][1] += 1  # Increase count
        else:
            face_count[sorted_face] = [face, 1]  # Store original face and set count to 1

    # Select faces which occur exactly once, preserving their original order.
    unique_faces = [original_face for original_face, count in face_count.values() if count == 1]

    return unique_faces

def add_chip_triangles(chip_thickness: float, chip_width: float, chip_length: float, fase_side_length: float, unit_conversion_factor: float) -> tuple[list, list, list]:
    '''
    Generate the triangles for the chip surrounding the channel network (except for the top face that includes the chip connections and the bottom face where the channel is open at the bottom side).
    And define the corner points of the chip.
    '''
    chip_triangles = []
    bottom_chip_triangles = []

    # define starting point of the module
    min_x = 0.0
    min_y = (7.5e-3 - 0.5 * chip_length * 1e-3) * unit_conversion_factor
    min_z = 0.0

    # Define clamp dimensions
    chip_radius_bottom = 0.5e-3 * unit_conversion_factor
    chip_radius = 1e-3 * unit_conversion_factor
    clamp_height = 1.5e-3 * unit_conversion_factor
    clamp_indent = 1e-3 * unit_conversion_factor

    # Chip edges definition
    vertice_0a = [min_x + fase_side_length + chip_radius, min_y + chip_radius, min_z]
    vertice_0b = [min_x + chip_radius, min_y + fase_side_length + chip_radius, min_z]
    vertice_1 =  [min_x + chip_width - chip_radius, min_y + chip_radius, min_z]
    vertice_2 =  [min_x + chip_width - chip_radius, min_y + chip_length - chip_radius, min_z]
    vertice_3 =  [min_x + chip_radius, min_y + chip_length - chip_radius, min_z]

    # Bottom corners Definition
    vertice_0_bottom = [min_x + clamp_indent + chip_radius_bottom, min_y + clamp_indent + chip_radius_bottom, min_z]
    vertice_1_bottom = [min_x + chip_width - clamp_indent - chip_radius_bottom, min_y + clamp_indent + chip_radius_bottom, min_z]
    vertice_2_bottom = [min_x + chip_width - clamp_indent - chip_radius_bottom, min_y + chip_length - clamp_indent - chip_radius_bottom, min_z] 
    vertice_3_bottom = [min_x  + clamp_indent + chip_radius_bottom, min_y + chip_length  - clamp_indent - chip_radius_bottom, min_z]


    vertice_0a_top = vertice_0a.copy()
    vertice_0a_top[2] += chip_thickness
    vertice_0b_top = vertice_0b.copy()
    vertice_0b_top[2] += chip_thickness
    vertice_1_top = vertice_1.copy()
    vertice_1_top[2] += chip_thickness
    vertice_2_top = vertice_2.copy()
    vertice_2_top[2] += chip_thickness
    vertice_3_top = vertice_3.copy()
    vertice_3_top[2] += chip_thickness

    # ADD THE ROUNDED CORNERS ON THE BOTTOM SIDE
    #There is no fase at the bottom
    vertice_0_bottom_start = [vertice_0_bottom[0] - chip_radius_bottom, vertice_0_bottom[1], vertice_0_bottom[2]]
    vertice_0_bottom_end = [vertice_0_bottom[0], vertice_0_bottom[1] - chip_radius_bottom, vertice_0_bottom[2]]
    arc_points_vertice_0 = discretize_arc(vertice_0_bottom_start, vertice_0_bottom, vertice_0_bottom_end, 0, 10, chip_radius_bottom, 0)
    center_points_bottom_0 = []
    for i in range(len(arc_points_vertice_0)):
        center_points_bottom_0 += [vertice_0_bottom]
    arc_triangles_bottom_0 = create_arc_triangles_xy(arc_points_vertice_0, center_points_bottom_0, 0.0, True)
    arc_triangles_bottom_0_connect, corner_point_0 = create_arc_triangles_xy_2(arc_points_vertice_0, center_points_bottom_0, clamp_height, False, 0)
    arc_triangles_bottom_side_0 = create_arc_triangles_side(arc_points_vertice_0, clamp_height, False)
    bottom_chip_triangles += arc_triangles_bottom_0 + arc_triangles_bottom_0_connect
    chip_triangles += arc_triangles_bottom_side_0

    vertice_1_bottom_start = [vertice_1_bottom[0] + chip_radius_bottom, vertice_1_bottom[1], vertice_1_bottom[2]]
    vertice_1_bottom_end = [vertice_1_bottom[0], vertice_1_bottom[1] - chip_radius_bottom, vertice_1_bottom[2]]
    arc_points_vertice_1 = discretize_arc(vertice_1_bottom_start, vertice_1_bottom, vertice_1_bottom_end, 0, 10, chip_radius_bottom, 0)
    center_points_bottom_1 = []
    for i in range(len(arc_points_vertice_1)):
        center_points_bottom_1 += [vertice_1_bottom]
    arc_triangles_bottom_1 = create_arc_triangles_xy(arc_points_vertice_1, center_points_bottom_1, 0, True)
    arc_triangles_bottom_1_connect, corner_point_1 = create_arc_triangles_xy_2(arc_points_vertice_1, center_points_bottom_1, clamp_height, False, 0)
    arc_triangles_bottom_side_1 = create_arc_triangles_side(arc_points_vertice_1, clamp_height, True)
    bottom_chip_triangles += arc_triangles_bottom_1 + arc_triangles_bottom_1_connect
    chip_triangles += arc_triangles_bottom_side_1

    vertice_2_bottom_start = [vertice_2_bottom[0] + chip_radius_bottom, vertice_2_bottom[1], vertice_2_bottom[2]]
    vertice_2_bottom_end = [vertice_2_bottom[0], vertice_2_bottom[1] + chip_radius_bottom, vertice_2_bottom[2]]
    arc_points_vertice_2 = discretize_arc(vertice_2_bottom_start, vertice_2_bottom, vertice_2_bottom_end, 0, 10, chip_radius_bottom, 0)
    center_points_bottom_2 = []
    for i in range(len(arc_points_vertice_2)):
        center_points_bottom_2 += [vertice_2_bottom]
    arc_triangles_bottom_2 = create_arc_triangles_xy(arc_points_vertice_2, center_points_bottom_2, 0.0, True)
    arc_triangles_bottom_2_connect , corner_point_2 = create_arc_triangles_xy_2(arc_points_vertice_2, center_points_bottom_2, clamp_height, False, 1)
    arc_triangles_bottom_side_2 = create_arc_triangles_side(arc_points_vertice_2, clamp_height, False)
    bottom_chip_triangles += arc_triangles_bottom_2 + arc_triangles_bottom_2_connect
    chip_triangles += arc_triangles_bottom_side_2

    vertice_3_bottom_start = [vertice_3_bottom[0] - chip_radius_bottom, vertice_3_bottom[1], vertice_3_bottom[2]]
    vertice_3_bottom_end = [vertice_3_bottom[0], vertice_3_bottom[1] + chip_radius_bottom, vertice_3_bottom[2]]
    arc_points_vertice_3 = discretize_arc(vertice_3_bottom_start, vertice_3_bottom, vertice_3_bottom_end, 1, 10, chip_radius_bottom, 0)
    center_points_bottom_3 = []
    for i in range(len(arc_points_vertice_3)):
        center_points_bottom_3 += [vertice_3_bottom]
    arc_triangles_bottom_3 = create_arc_triangles_xy(arc_points_vertice_3, center_points_bottom_3, 0.0, True)
    arc_triangles_bottom_3_connect , corner_point_3 = create_arc_triangles_xy_2(arc_points_vertice_3, center_points_bottom_3, clamp_height, False, 1)
    arc_triangles_bottom_side_3 = create_arc_triangles_side(arc_points_vertice_3, clamp_height, True)
    bottom_chip_triangles += arc_triangles_bottom_3 + arc_triangles_bottom_3_connect
    chip_triangles += arc_triangles_bottom_side_3

    # DEFINE THE CONNECTION BETWEEN THE ROUNDED EDGES ON THE BOTTOM SIDE
    bottom_chip_triangles += [
        [vertice_0_bottom, vertice_0_bottom_start, vertice_3_bottom],
        [vertice_3_bottom, vertice_0_bottom_start, vertice_3_bottom_start],
        [vertice_0_bottom_end, vertice_0_bottom, vertice_1_bottom],
        [vertice_0_bottom_end, vertice_1_bottom, vertice_1_bottom_end],
        [vertice_1_bottom_start, vertice_1_bottom, vertice_2_bottom],
        [vertice_1_bottom_start, vertice_2_bottom, vertice_2_bottom_start],
        [vertice_2_bottom_end, vertice_2_bottom, vertice_3_bottom],
        [vertice_2_bottom_end, vertice_3_bottom, vertice_3_bottom_end]
    ]

    # DEFINE THE SIDES OF THE CLAMP
    chip_triangles += [
        [vertice_3_bottom_start, vertice_0_bottom_start, [vertice_3_bottom_start[0], vertice_3_bottom_start[1], vertice_3_bottom_start[2] + clamp_height]],
        [[vertice_3_bottom_start[0], vertice_3_bottom_start[1], vertice_3_bottom_start[2] + clamp_height], vertice_0_bottom_start, [vertice_0_bottom_start[0], vertice_0_bottom_start[1], vertice_0_bottom_start[2] + clamp_height]],
        [vertice_0_bottom_end, vertice_1_bottom_end, [vertice_1_bottom_end[0], vertice_1_bottom_end[1], vertice_1_bottom_end[2] + clamp_height]],
        [vertice_0_bottom_end, [vertice_1_bottom_end[0], vertice_1_bottom_end[1], vertice_1_bottom_end[2] + clamp_height], [vertice_0_bottom_end[0], vertice_0_bottom_end[1], vertice_0_bottom_end[2] + clamp_height]],
        [vertice_1_bottom_start, vertice_2_bottom_start, [vertice_2_bottom_start[0], vertice_2_bottom_start[1], vertice_2_bottom_start[2] + clamp_height]],
        [vertice_1_bottom_start, [vertice_2_bottom_start[0], vertice_2_bottom_start[1], vertice_2_bottom_start[2] + clamp_height], [vertice_1_bottom_start[0], vertice_1_bottom_start[1], vertice_1_bottom_start[2] + clamp_height]],
        [vertice_2_bottom_end, vertice_3_bottom_end, [vertice_3_bottom_end[0], vertice_3_bottom_end[1], vertice_3_bottom_end[2] + clamp_height]],
        [vertice_2_bottom_end, [vertice_3_bottom_end[0], vertice_3_bottom_end[1], vertice_3_bottom_end[2] + clamp_height], [vertice_2_bottom_end[0], vertice_2_bottom_end[1], vertice_2_bottom_end[2] + clamp_height]]
    ]

    # DEFINE THE ARCS FROM THE TOP LAYER
    # fase_rounded_length = math.sqrt(2 * (fase_side_length**2) - 4 * fase_side_length * math.cos(45)) * math.sin(45)
    fase_1_rounded_length = 0.29287e-3 * unit_conversion_factor # TODO calculate based on input values
    fase_2_rounded_length = 0.70705e-3 * unit_conversion_factor

    vertice_0a_top_start = [vertice_0a_top[0] - fase_2_rounded_length, vertice_0a_top[1]  - chip_radius + fase_1_rounded_length, vertice_0a_top[2]]
    vertice_0a_top_end = [vertice_0a_top[0], vertice_0a_top[1] - chip_radius, vertice_0a_top[2]]
    arc_points_vertice_0a = discretize_arc_2(vertice_0a_top_start, vertice_0a_top, vertice_0a_top_end, 0, 10, chip_radius, 0)
    center_points_top_0a = []
    for i in range(len(arc_points_vertice_0a)):
        center_points_top_0a += [vertice_0a_top]
    arc_triangles_top_0a = create_arc_triangles_xy(arc_points_vertice_0a, center_points_top_0a, min_z, True)
    bottom_chip_triangles += create_arc_triangles_xy(arc_points_vertice_0a, center_points_top_0a, clamp_height - chip_thickness, True)
    arc_triangles_top_side_0a = create_arc_triangles_side(arc_points_vertice_0a, clamp_height - chip_thickness, True)
    chip_triangles += arc_triangles_top_side_0a + arc_triangles_top_0a # + arc_triangles_top_0a_connect

    vertice_0b_top_start = [vertice_0b_top[0] - chip_radius, vertice_0b_top[1], vertice_0b_top[2]]
    vertice_0b_top_end = [vertice_0b_top[0] - fase_2_rounded_length, vertice_0b_top[1] - chip_radius + fase_1_rounded_length, vertice_0b_top[2]]
    arc_points_vertice_0b = discretize_arc_2(vertice_0b_top_start, vertice_0b_top, vertice_0b_top_end, -2*np.pi, 10, chip_radius, 0)
    center_points_top_0b = []
    for i in range(len(arc_points_vertice_0b)):
        center_points_top_0b += [vertice_0b_top]
    arc_triangles_top_0b = create_arc_triangles_xy(arc_points_vertice_0b, center_points_top_0b, min_z, True)
    bottom_chip_triangles += create_arc_triangles_xy(arc_points_vertice_0b, center_points_top_0b, clamp_height - chip_thickness, True)
    arc_triangles_top_side_0b = create_arc_triangles_side(arc_points_vertice_0b, clamp_height - chip_thickness, True)
    chip_triangles += arc_triangles_top_0b + arc_triangles_top_side_0b # + arc_triangles_top_0b_connect

    vertice_1_top_start = [vertice_1_top[0] + chip_radius, vertice_1_top[1], vertice_1_top[2]]
    vertice_1_top_end = [vertice_1_top[0], vertice_1_top[1] - chip_radius, vertice_1_top[2]]
    arc_points_vertice_1 = discretize_arc_2(vertice_1_top_start, vertice_1_top, vertice_1_top_end, 0, 10, chip_radius, 0)
    center_points_top_1 = []
    for i in range(len(arc_points_vertice_1)):
        center_points_top_1 += [vertice_1_top]
    arc_triangles_top_1 = create_arc_triangles_xy(arc_points_vertice_1, center_points_top_1, min_z, True)
    bottom_chip_triangles += create_arc_triangles_xy(arc_points_vertice_1, center_points_top_1, clamp_height - chip_thickness, True)
    arc_triangles_top_side_1 = create_arc_triangles_side(arc_points_vertice_1, clamp_height - chip_thickness, False)
    chip_triangles += arc_triangles_top_1 + arc_triangles_top_side_1 #+ arc_triangles_top_1_connect

    vertice_2_top_start = [vertice_2_top[0] + chip_radius, vertice_2_top[1], vertice_2_top[2]]
    vertice_2_top_end = [vertice_2_top[0], vertice_2_top[1] + chip_radius, vertice_2_top[2]]
    arc_points_vertice_2 = discretize_arc_2(vertice_2_top_start, vertice_2_top, vertice_2_top_end, 0, 10, chip_radius, 0)
    center_points_top_2 = []
    for i in range(len(arc_points_vertice_2)):
        center_points_top_2 += [vertice_2_top]
    arc_triangles_top_2 = create_arc_triangles_xy(arc_points_vertice_2, center_points_top_2, min_z, True)
    bottom_chip_triangles += create_arc_triangles_xy(arc_points_vertice_2, center_points_top_2, clamp_height - chip_thickness, True)
    arc_triangles_top_side_2 = create_arc_triangles_side(arc_points_vertice_2, clamp_height - chip_thickness, True)
    chip_triangles += arc_triangles_top_2 + arc_triangles_top_side_2 # + arc_triangles_top_2_connect

    vertice_3_top_start = [vertice_3_top[0] - chip_radius, vertice_3_top[1], vertice_3_top[2]]
    vertice_3_top_end = [vertice_3_top[0], vertice_3_top[1] + chip_radius, vertice_3_top[2]]
    arc_points_vertice_3 = discretize_arc_2(vertice_3_top_start, vertice_3_top, vertice_3_top_end, 0, 10, chip_radius, 0)
    center_points_top_3 = []
    for i in range(len(arc_points_vertice_3)):
        center_points_top_3 += [vertice_3_top]
    arc_triangles_top_3 = create_arc_triangles_xy(arc_points_vertice_3, center_points_top_3, min_z, True)
    bottom_chip_triangles += create_arc_triangles_xy(arc_points_vertice_3, center_points_top_3, clamp_height - chip_thickness, True)
    arc_triangles_top_side_3 = create_arc_triangles_side(arc_points_vertice_3, clamp_height - chip_thickness, False)
    chip_triangles += arc_triangles_top_3 + arc_triangles_top_side_3 # + arc_triangles_top_3_connect

    # Define triangles for the clamp top layer (in the xy-plane)
    chip_triangles += [
        [vertice_0b_top_start, vertice_0b_top, vertice_3_top],
        [vertice_0b_top_start, vertice_3_top, vertice_3_top_start],
        [vertice_0a_top, vertice_0a_top_end, vertice_1_top],
        [vertice_1_top, vertice_0a_top_end, vertice_1_top_end],
        [vertice_1_top, vertice_1_top_start, vertice_2_top],
        [vertice_2_top, vertice_1_top_start, vertice_2_top_start],
        [vertice_2_top, vertice_2_top_end, vertice_3_top],
        [vertice_3_top, vertice_2_top_end, vertice_3_top_end]
    ]

    chip_triangles += [
        [vertice_0a_top, vertice_0b_top, vertice_0a_top_start],
        [vertice_0a_top_start, vertice_0b_top, vertice_0b_top_end]
    ]
    bottom_chip_triangles += [
        [corner_point_0, [vertice_0a_top[0], vertice_0a_top[1], vertice_0a_top[2] + clamp_height - chip_thickness], [vertice_0a_top_start[0], vertice_0a_top_start[1], vertice_0a_top_start[2] + clamp_height - chip_thickness]],
        [[vertice_0b_top[0], vertice_0b_top[1], vertice_0b_top[2] + clamp_height - chip_thickness], corner_point_0, [vertice_0b_top_end[0], vertice_0b_top_end[1], vertice_0b_top_end[2] + clamp_height - chip_thickness]],
        [corner_point_0, [vertice_0a_top_start[0], vertice_0a_top_start[1], vertice_0a_top_start[2] + clamp_height - chip_thickness], [vertice_0b_top_end[0], vertice_0b_top_end[1], vertice_0b_top_end[2] + clamp_height - chip_thickness]]
    ]

    # CLOSE THE FASE AT THE SIDE
    chip_triangles += [
        [vertice_0a_top_start, vertice_0b_top_end, [vertice_0a_top_start[0], vertice_0a_top_start[1], vertice_0a_top_start[2] + clamp_height - chip_thickness]],
        [[vertice_0a_top_start[0], vertice_0a_top_start[1], vertice_0a_top_start[2] + clamp_height - chip_thickness], vertice_0b_top_end, [vertice_0b_top_end[0], vertice_0b_top_end[1], vertice_0b_top_end[2] + clamp_height - chip_thickness]]
    ]
        
    # DEFINE THE SIDES OF THE CHIP
    chip_triangles += [
        [vertice_0b_top_start, vertice_3_top_start, [vertice_3_top_start[0], vertice_3_top_start[1], vertice_3_top_start[2] + clamp_height - chip_thickness]],
        [vertice_0b_top_start, [vertice_3_top_start[0], vertice_3_top_start[1], vertice_3_top_start[2] + clamp_height - chip_thickness], [vertice_0b_top_start[0], vertice_0b_top_start[1], vertice_0b_top_start[2] + clamp_height - chip_thickness]],
        [vertice_1_top_end, vertice_0a_top_end, [vertice_1_top_end[0], vertice_1_top_end[1], vertice_1_top_end[2] + clamp_height - chip_thickness]],
        [[vertice_1_top_end[0], vertice_1_top_end[1], vertice_1_top_end[2] + clamp_height - chip_thickness], vertice_0a_top_end, [vertice_0a_top_end[0], vertice_0a_top_end[1], vertice_0a_top_end[2] + clamp_height - chip_thickness]],
        [vertice_2_top_start, vertice_1_top_start, [vertice_2_top_start[0], vertice_2_top_start[1], vertice_2_top_start[2] + clamp_height - chip_thickness]],
        [[vertice_2_top_start[0], vertice_2_top_start[1], vertice_2_top_start[2] + clamp_height - chip_thickness], vertice_1_top_start, [vertice_1_top_start[0], vertice_1_top_start[1], vertice_1_top_start[2] + clamp_height - chip_thickness]],
        [vertice_3_top_end, vertice_2_top_end, [vertice_3_top_end[0], vertice_3_top_end[1], vertice_3_top_end[2] + clamp_height - chip_thickness]],
        [[vertice_3_top_end[0], vertice_3_top_end[1], vertice_3_top_end[2] + clamp_height - chip_thickness], vertice_2_top_end, [vertice_2_top_end[0], vertice_2_top_end[1], vertice_2_top_end[2] + clamp_height - chip_thickness]]
    ]

    # DEFINE THE MIDDLE LAYER OF THE CHIP
    bottom_chip_triangles += [
        [[corner_point_0[0] + chip_radius, corner_point_0[1], corner_point_0[2]], corner_point_1, [vertice_0a_top_end[0], vertice_0a_top_end[1], vertice_0a_top_end[2] + clamp_height - chip_thickness]],
        [[vertice_1_top_end[0], vertice_1_top_end[1], vertice_1_top_end[2] + clamp_height - chip_thickness], [vertice_0a_top_end[0], vertice_0a_top_end[1], vertice_0a_top_end[2] + clamp_height - chip_thickness], corner_point_1],
        [corner_point_1, corner_point_2, [vertice_1_top_start[0], vertice_1_top_start[1], vertice_1_top_start[2] + clamp_height - chip_thickness]],
        [[vertice_2_top_start[0], vertice_2_top_start[1], vertice_2_top_start[2] + clamp_height - chip_thickness], [vertice_1_top_start[0], vertice_1_top_start[1], vertice_1_top_start[2] + clamp_height - chip_thickness], corner_point_2],
        [corner_point_2, corner_point_3, [vertice_2_top_end[0], vertice_2_top_end[1], vertice_2_top_end[2] + clamp_height - chip_thickness]],
        [[vertice_3_top_end[0], vertice_3_top_end[1], vertice_3_top_end[2] + clamp_height - chip_thickness], [vertice_2_top_end[0], vertice_2_top_end[1], vertice_2_top_end[2] + clamp_height - chip_thickness], corner_point_3],
        [corner_point_3, [corner_point_0[0], corner_point_0[1] + chip_radius, corner_point_0[2]], [vertice_3_top_start[0], vertice_3_top_start[1], vertice_3_top_start[2] + clamp_height - chip_thickness]],
        [[vertice_0b_top_start[0], vertice_0b_top_start[1], vertice_0b_top_start[2] + clamp_height - chip_thickness], [vertice_3_top_start[0], vertice_3_top_start[1], vertice_3_top_start[2] + clamp_height - chip_thickness], [corner_point_0[0], corner_point_0[1] + chip_radius, corner_point_0[2]]]
    ]

    chip_top_corners = [vertice_0a_top, vertice_0b_top, vertice_1_top, vertice_2_top, vertice_3_top]
    chip_bottom_corners = [vertice_0_bottom, vertice_0_bottom, vertice_1_bottom, vertice_2_bottom, vertice_3_bottom] # TODO

    return chip_triangles, bottom_chip_triangles, chip_top_corners, chip_bottom_corners

def add_chip_top_layer_triangles(chip_top_corners: list, corner_points: list) -> list:
    '''
    The top layer of the chip definition houses the luer connections of the chip, this needs to be considered when defining the triangles as to not close off the connections.
    The function returns the triangles for the top layer of the chip.
    '''
    triangles = []

    # Chip Top Layer
    vertice_0a = chip_top_corners[0] # this includes the fase
    vertice_0b = chip_top_corners[1]
    vertice_1 = chip_top_corners[2]
    vertice_2 = chip_top_corners[3]
    vertice_3 = chip_top_corners[4] 
    

    # luer Inlet and Outlet, these are not defined the same way all other faces are defined
    #  In:  1-----3        Out: 3-----1          
    #       |     |             |     | 
    #       0-----2             2-----0
    inlet_corner_points = corner_points[:4]
    outlet_corner_points = corner_points[4:8]

    # Corners to inlet and outlet
    triangles += [
        [inlet_corner_points[0], vertice_0a, inlet_corner_points[2]],
        [inlet_corner_points[2], vertice_0a, outlet_corner_points[2]],
        [outlet_corner_points[2], vertice_0a, vertice_1],
        [vertice_0a, inlet_corner_points[0], inlet_corner_points[0]],
        [vertice_0a, inlet_corner_points[0], vertice_0b],
        [outlet_corner_points[2], vertice_1, outlet_corner_points[0]],
        [outlet_corner_points[0], vertice_1, outlet_corner_points[1]],
        [outlet_corner_points[1], vertice_1, vertice_2],
        [vertice_3, vertice_0b, inlet_corner_points[0]],
        [vertice_3, inlet_corner_points[1], vertice_2],
        [outlet_corner_points[1], vertice_2, outlet_corner_points[3]],
        [outlet_corner_points[3], vertice_2, inlet_corner_points[3]],
        [inlet_corner_points[3], vertice_2, inlet_corner_points[1]],
        [inlet_corner_points[2], outlet_corner_points[2], outlet_corner_points[3]],
        [inlet_corner_points[2], outlet_corner_points[3], inlet_corner_points[3]],
        [vertice_3, inlet_corner_points[0], inlet_corner_points[1]]         
    ]
    
    return triangles

def add_chip_bottom_layer_triangles(chip_bottom_corners: list, corner_points_via: list, corner_points_trap: list, corner_points_arc: list, channel_width: float, nr_of_traps: int) -> list:
    '''
    The bottom layer of the chip includes the open bottom side of the channel network to facilitate fabrication. 
    For this the channel network, trap definition, and via defintion (and corner points) are taken into account and the the chip is closed off at the bottom using triangles.
    The definition accounts for the number of traps that are included, and the rounded edges at the side of the chip.
    '''
    triangles = []

    trap_width = channel_width * 3/8 # TODO this only works as long as all the traps have the same width, otherwise this needs to be based on each trap left to right 
    trap_distance = (corner_points_trap[0][1] - corner_points_trap[1][1] - trap_width) / 2

    for i in range(len(chip_bottom_corners)):
        chip_bottom_corners[i][2] = 0.0

    vertice_0a = chip_bottom_corners[0]
    vertice_0b = chip_bottom_corners[1]
    vertice_1 = chip_bottom_corners[2]
    vertice_2 = chip_bottom_corners[3]
    vertice_3 = chip_bottom_corners[4]

    inlet_points = corner_points_via[:4]
    outlet_points = corner_points_via[4:8]

    triangles += [ # Outlet via to side
        [outlet_points[3], vertice_2, vertice_1],
        [outlet_points[3], outlet_points[2], vertice_2],
        [outlet_points[1], outlet_points[3], vertice_1],
        [outlet_points[1], vertice_1, outlet_points[0]]]
    
    triangles += [ # Inlet via to side
        [vertice_0a, inlet_points[1], inlet_points[0]],
        [vertice_0a, inlet_points[3], inlet_points[1]],
        [vertice_3, inlet_points[3], vertice_0a],
        [vertice_3, inlet_points[2], inlet_points[3]]
    ]

    triangles += [ # Arcs Right/Bottom Side
        [corner_points_arc[7 + nr_of_traps * 8], vertice_2, outlet_points[2]],
        [vertice_2, corner_points_arc[7 + nr_of_traps * 8], corner_points_arc[4 + nr_of_traps * 8]],
        [vertice_2, corner_points_arc[4 + nr_of_traps * 8], corner_points_arc[2 + nr_of_traps * 8]],
        [corner_points_arc[4], vertice_2, corner_points_arc[10]],
        [corner_points_arc[2], vertice_2, corner_points_arc[4]],
        [vertice_2, corner_points_arc[2], vertice_3],
        # [vertice_3, corner_points_arc[2], vertice_3],
        [vertice_3, corner_points_arc[2], corner_points_arc[1]],
        [vertice_3, corner_points_arc[1], inlet_points[2]]
    ]

    triangles += [ # Arcs Left/Top Side
        [vertice_1, corner_points_arc[6 + nr_of_traps * 8], outlet_points[0]],
        [corner_points_arc[14], vertice_1, corner_points_arc[8]],
        [vertice_1, corner_points_arc[6], corner_points_arc[8]],
        [vertice_1, vertice_0a, corner_points_arc[6]],
        [corner_points_arc[6], vertice_0a, corner_points_arc[0]],
        [corner_points_arc[0], vertice_0a, inlet_points[0]],
        [vertice_0b, vertice_0a, vertice_3]
    ]

    for i in range(1, nr_of_traps):
        triangles += [ # Arcs Right/Bottom Side
            [vertice_2, corner_points_arc[4 + i * 8], corner_points_arc[2 + i * 8]],
            [vertice_2, corner_points_arc[10 + i * 8], corner_points_arc[4 + i * 8]],
        ]
        triangles += [ # Arcs Left/Top Side
            [vertice_1, corner_points_arc[6 + i * 8], corner_points_arc[8 + i * 8]],
            [vertice_1, corner_points_arc[8 + i * 8], corner_points_arc[14 + i * 8]]
        ]

    for i in range(nr_of_traps):
        triangles += [ # Trap
            # Bottom
            [corner_points_arc[4 + i * 8], corner_points_arc[10 + i * 8], corner_points_trap[0 + 2 * i]],
            [corner_points_arc[4 + i * 8], corner_points_trap[0 + 2 * i], np.array([corner_points_arc[4 + i * 8][0], corner_points_trap[0 + 2 * i][1], corner_points_arc[4 + i * 8][2]], dtype=np.float64)], 
            [corner_points_trap[0 + 2 * i], corner_points_arc[10 + i * 8], np.array([corner_points_arc[10 + i * 8][0], corner_points_trap[0 + 2 * i][1], corner_points_arc[10 + i * 8][2]], dtype=np.float64)], 
            [np.array([corner_points_trap[0 + 2 * i][0], corner_points_trap[0 + 2 * i][1] - trap_distance, corner_points_trap[0 + 2 * i][2]], dtype=np.float64), corner_points_trap[0 + 2 * i], np.array([corner_points_arc[10 + i * 8][0], corner_points_trap[0 + 2 * i][1] - trap_distance, corner_points_arc[10 + i * 8][2]], dtype=np.float64)],
            [corner_points_trap[0 + 2 * i], np.array([corner_points_arc[10 + i * 8][0], corner_points_trap[0 + 2 * i][1], corner_points_arc[10 + i * 8][2]], dtype=np.float64), np.array([corner_points_arc[10 + i * 8][0], corner_points_trap[0 + 2 * i][1] - trap_distance, corner_points_arc[10][2]], dtype=np.float64)],
            # Top
            [corner_points_trap[1 + 2 * i], np.array([corner_points_arc[4 + i * 8][0], corner_points_arc[7 + i * 8][1], corner_points_arc[7 + i * 8][2]], dtype=np.float64), np.array([corner_points_arc[4 + i * 8][0], corner_points_trap[1 + 2 * i][1], corner_points_arc[4 + i * 8][2]], dtype=np.float64)],
            [np.array([corner_points_arc[4 + i * 8][0], corner_points_arc[7 + i * 8][1], corner_points_arc[7 + i * 8][2]], dtype=np.float64), corner_points_trap[1 + 2 * i], np.array([corner_points_arc[10 + i * 8][0], corner_points_arc[7 + i * 8][1], corner_points_arc[4 + i * 8][2]], dtype=np.float64)],
            [np.array([corner_points_arc[10 + i * 8][0], corner_points_arc[7 + i * 8][1], corner_points_arc[7 + i * 8][2]], dtype=np.float64), corner_points_trap[1 + 2 * i], np.array([corner_points_arc[10 + i * 8][0], corner_points_trap[1 + 2 * i][1], corner_points_arc[10 + i * 8][2]], dtype=np.float64)],
            [corner_points_trap[1 + 2 * i], np.array([corner_points_trap[1 + 2 * i][0], corner_points_trap[1 + 2 * i][1] + trap_distance, corner_points_trap[1 + 2 * i][2]], dtype=np.float64), np.array([corner_points_arc[10 + i * 8][0], corner_points_trap[1 + 2 * i][1] + trap_distance, corner_points_trap[1 + 2 * i][2]], dtype=np.float64)],
            [corner_points_trap[1 + 2 * i], np.array([corner_points_arc[10 + i * 8][0], corner_points_trap[1 + 2 * i][1] + trap_distance, corner_points_trap[1 + 2 * i][2]], dtype=np.float64), np.array([corner_points_arc[10 + i * 8][0], corner_points_trap[1 + 2 * i][1], corner_points_trap[1 + 2 * i][2]], dtype=np.float64)]
        ]

    via_distance = (inlet_points[2][1] - inlet_points[0][1] - channel_width) / 2

    triangles += [ # extra connections to fill the gaps
        # Inlet
        [corner_points_arc[0], inlet_points[0], np.array([inlet_points[0][0], corner_points_arc[0][1], corner_points_arc[0][2]], dtype=np.float64)],
        [corner_points_arc[1], corner_points_arc[2], np.array([corner_points_arc[2][0], corner_points_arc[1][1], corner_points_arc[1][2]], dtype=np.float64)],
        [inlet_points[2], corner_points_arc[1], np.array([inlet_points[2][0], inlet_points[2][1] - via_distance, inlet_points[2][2]], dtype=np.float64)],
        [corner_points_arc[1], np.array([corner_points_arc[1][0], inlet_points[2][1] - via_distance, inlet_points[2][2]], dtype=np.float64), np.array([inlet_points[2][0], inlet_points[2][1] - via_distance, inlet_points[2][2]], dtype=np.float64)],
        # First Meander
        [corner_points_arc[6], corner_points_arc[0], np.array([corner_points_arc[6][0], corner_points_arc[0][1], corner_points_arc[0][2]], dtype=np.float64)],
        [np.array([corner_points_arc[6][0], corner_points_arc[0][1], corner_points_arc[0][2]], dtype=np.float64), corner_points_arc[0], np.array([corner_points_arc[6][0], corner_points_arc[3][1], corner_points_arc[2][2]], dtype=np.float64)],
        [corner_points_arc[0], np.array([corner_points_arc[0][0], corner_points_arc[3][1], corner_points_arc[3][2]], dtype=np.float64), np.array([corner_points_arc[6][0], corner_points_arc[3][1], corner_points_arc[2][2]], dtype=np.float64)],
        # # Last Meander
        [corner_points_arc[6 + nr_of_traps * 8], corner_points_arc[nr_of_traps * 8], np.array([corner_points_arc[nr_of_traps * 8][0], corner_points_arc[6 + nr_of_traps * 8][1], corner_points_arc[6 + nr_of_traps * 8][2]], dtype=np.float64)],
        [corner_points_arc[6 + nr_of_traps * 8], np.array([corner_points_arc[nr_of_traps * 8][0], corner_points_arc[6 + nr_of_traps * 8][1], corner_points_arc[6 + nr_of_traps * 8][2]], dtype=np.float64), np.array([corner_points_arc[6 + nr_of_traps * 8][0], corner_points_arc[5 + nr_of_traps * 8][1], corner_points_arc[5 + nr_of_traps * 8][2]], dtype=np.float64)],
        [np.array([corner_points_arc[nr_of_traps * 8][0], corner_points_arc[6 + nr_of_traps * 8][1], corner_points_arc[6 + nr_of_traps * 8][2]], dtype=np.float64), np.array([corner_points_arc[nr_of_traps * 8][0], corner_points_arc[5 + nr_of_traps * 8][1], corner_points_arc[5 + nr_of_traps * 8][2]], dtype=np.float64), np.array([corner_points_arc[6 + nr_of_traps * 8][0], corner_points_arc[3 + nr_of_traps * 8][1], corner_points_arc[3 + nr_of_traps * 8][2]], dtype=np.float64)],
        # Outlet
        [outlet_points[0], corner_points_arc[6 + nr_of_traps * 8], np.array([outlet_points[0][0], corner_points_arc[6 + nr_of_traps * 8][1], corner_points_arc[6 + nr_of_traps * 8][2]], dtype=np.float64)],
        [corner_points_arc[4 + nr_of_traps * 8], corner_points_arc[7 + nr_of_traps * 8], np.array([corner_points_arc[4 + nr_of_traps * 8][0], corner_points_arc[7 + nr_of_traps * 8][1], corner_points_arc[7 + nr_of_traps * 8][2]], dtype=np.float64)],
        [corner_points_arc[7 + nr_of_traps * 8], outlet_points[2], np.array([outlet_points[2][0], outlet_points[2][1] - via_distance, outlet_points[2][2]], dtype=np.float64)],
        [np.array([corner_points_arc[7 + nr_of_traps * 8][0], outlet_points[2][1] - via_distance, outlet_points[2][2]], dtype=np.float64), corner_points_arc[7 + nr_of_traps * 8], np.array([outlet_points[2][0], outlet_points[2][1] - via_distance, outlet_points[2][2]], dtype=np.float64)]
    ] 
    
    for i in range(1, nr_of_traps):
        triangles += [ 
            # Extra Meander if there is more than one trap
            [corner_points_arc[6 + i * 8], corner_points_arc[i * 8], np.array([corner_points_arc[i * 8][0], corner_points_arc[3 + i * 8][1], corner_points_arc[i * 8][2]], dtype=np.float64)], 
            [np.array([corner_points_arc[6 + i * 8][0], corner_points_arc[5 + i * 8][1], corner_points_arc[6 + i *8][2]], dtype=np.float64), corner_points_arc[6 + i * 8], np.array([corner_points_arc[i * 8][0], corner_points_arc[3 + i * 8][1], corner_points_arc[i * 8][2]], dtype=np.float64)],
        ]

    return triangles

def create_arc_triangles_xy_2(arc_points: list, arc_points2: list, height: float, luer_type: bool, direction: float) -> list:
    '''
    Creates the triangles for the outer chip layer (i.e., the top and bottom layer). Here the triangles face outward. 
    This allows to connect the round openings for the tubing, luer connection or the open channel face to be connected to the closed off chip layers.
    '''
    triangles = []
    
    for i in range(len(arc_points) - 1):
        p1A, p2A = np.array(arc_points[i], dtype=np.float64), np.array(arc_points[i + 1], dtype=np.float64)
        p1B, p2B = np.array(arc_points2[i], dtype=np.float64), np.array(arc_points2[i + 1], dtype=np.float64)

        segment_vec = p2A - p1A
        segment_length = np.linalg.norm(segment_vec)

        # Skip zero-length segments
        if segment_length == 0:
            continue

        # Define surrounding box to be able to triangulate the top layer of the chip correctly
        p1A_top = p1A.copy()
        p1A_top[2] += height
        p2A_top = p2A.copy()
        p2A_top[2] += height

        # create a connection from the arc points to the side (in either +x or -x direction)
        nr_end_point = len(arc_points)-1
        delta_x = np.array(arc_points[0], dtype=np.float64)[0] - np.array(arc_points[nr_end_point], dtype=np.float64)[0]
        delta_y = np.array(arc_points[0], dtype=np.float64)[1] - np.array(arc_points[nr_end_point], dtype=np.float64)[1]

        p_corner = np.array(arc_points[0])
        if luer_type:
            p_corner[0] -= (delta_x)
        else:
            p_corner[1] -= (delta_y)
        p_corner[2] += height

        triangles.append([tuple(p1A_top), tuple(p2A_top), tuple(p_corner)]) # this works for the luer inlet and outlet

    corner_point = tuple(p_corner)

    return triangles, corner_point

def enlarge(points: list, center: list, distance: float) -> list:
    '''
    Enlarges the points around the center point by a given distance.
    This is required to include the angled luer connections.
    '''
    enlarged_points = []
    for point in points:
        vector = np.array(point, dtype=np.float64) - np.array(center, dtype=np.float64)
        vector_length = np.linalg.norm(vector)
        if vector_length == 0:
            # Avoid division by zero for the center point
            enlarged_points.append(point)
        else:
            normalized_vector = vector / vector_length
            new_point = np.array(center, dtype=np.float64) + (vector_length + distance) * normalized_vector
            enlarged_points.append(new_point.tolist())
    return enlarged_points

def distance_3d(p1: list, p2: list) -> float:
    '''
    Calculates the distance between two points.
    '''
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

def find_overlap_xy(p1: list, p2: list, p3: list, p4: list) -> list:
    '''
    Find the overlapping point between two lines.
    '''
    # Line AB represented as a1x + b1y = c1
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]
    
    # Line CD represented as a2x + b2y = c2
    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2 * p3[0] + b2 * p3[1]
    
    determinant = a1 * b2 - a2 * b1
    
    if determinant == 0:
        # Lines are parallel, no intersection
        return None
    else:
        # Calculate the intersection point
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
    return [x, y, p1[2]]


def define_luer_arcs(height: float, start: list, end: list, direction: list, numSegments: int, direction2: float, luer_radius: float, luer_angle: float, channel_height: float) -> tuple[list, list, list]:
    '''
    Define the triangles, corner, and arc points for the luer connection.
    '''
    triangles = []
    center_points = []

    step = -channel_height
    radius = 0.5 * distance_3d(start, end) * direction2
    center = 0.5 * (np.array(start) + np.array(end)) # TODO add the higher precision here
    # center = 0.5 * (start + end)
    arcDirection = 0

    if direction[0] != 0:
            luer_type = True #Inlet/Outlet luer
            center_new = center.copy()
            center_new[0] -= step * 0.5 * direction[0]

            p1 = [center_new[0] - radius, center_new[1], center_new[2]]

            start_new = start.copy()
            start_new[0] -= step * 0.5 * direction[0]
    
            if math.isclose(p1[0], start_new[0] - abs(radius), rel_tol=eps) and math.isclose(p1[1], start_new[1] + abs(radius), rel_tol=eps): 
                arcDirection = 2 * np.pi

    arc_points = discretize_arc_2(start_new, center_new, p1, arcDirection, numSegments, luer_radius, 0)
    luer_height = height
    angle_chamfer_luer = (luer_angle - 90) * math.pi/180 # TODO make this dependent on mini luer vs luer, dependent on the angle
    distance_chamfer_luer = math.tan(angle_chamfer_luer) * luer_height

    arc_points_top = enlarge(arc_points, center_new, distance_chamfer_luer)

    arc_triangles_side = create_arc_triangles_side_angled(arc_points, arc_points_top, luer_height)

    for i in range(numSegments+1):
        center_points.append([center_new[0], center_new[1], center_new[2]])

    arc_triangles_xy, corner_point = create_arc_triangles_xy_2(arc_points_top, center_points, height, luer_type, 1)

    triangles.extend(arc_triangles_side)
    triangles.extend(arc_triangles_xy)

    return triangles, corner_point, arc_points

def add_luer_connections(luer_vertices: list, vertices: list, direction: list, numSegments: int, chip_height_top: float, luer_radius: float, luer_angle: float, channel_height: float, quad_list: list, nr_of_traps: int, unit_conversion_factor) -> tuple[list, list, list]:
    '''
    Creates the luer connections from the ground faces to the chip.
    Includes a cylinder for each luer connection to be able to connect the tubes for the luers.
    This cylinder is connected to the chip by another cylynder.
    (The alternative rounded edge that is connected to the inlet cylinder via an intermediate section is commented out since it is more difficult to fabricate.)
    '''

    triangles = []
    bottom_triangles = []
    top_corner_points = [] # to define the top face of the chip

    channel_corner_points = [] # define the corner points of the intermediate rounded channel connection
    via_triangles = []
    all_arc_points = []

    # 1. VIA CONNECTIONS
    via_height = channel_height # this is the minimal length of the luer
    height = chip_height_top - via_height

    # Define the connection based on a flat cylinder 
    via_arc_points = []
    via_center_points = []
    via_corner_points = []

    via_radius = min(1.25 * channel_height, luer_radius - eps) # because channel_height = channel_width

    distance_channel_circle = (via_radius - math.sqrt(via_radius ** 2 - (channel_height/2) ** 2))

    via_center = [luer_vertices[0][0] + (via_radius - distance_channel_circle) * direction[0], luer_vertices[0][1] + 0.5 * channel_height, luer_vertices[0][2]]
    for i in range(numSegments+1):
        via_center_points.append(via_center)
    
    end1 = [luer_vertices[0][0] + (via_radius - distance_channel_circle) * direction[0], luer_vertices[0][1] + (0.5 * channel_height - via_radius) * direction[0], luer_vertices[0][2]]  
    end2 = [luer_vertices[0][0] + (via_radius * 2 - distance_channel_circle) * direction[0], luer_vertices[0][1] + 0.5 * channel_height, luer_vertices[0][2]]  

    direction2 = 0
    if direction[0] == -1: # Inlet (left side of the chip)
        direction2 = np.pi
    
    # Arc 1
    arc_points_1 = discretize_arc_2(luer_vertices[0], via_center, end1, direction2, numSegments, via_radius * direction[0], channel_height)
    arc_triangles_side = create_arc_triangles_side(arc_points_1, channel_height, inner_arc=True) # TODO
    bottom_via_triangles, via_corner_points_1 = create_arc_triangles_xy_2(arc_points_1, via_center_points, 0.0, False, via_radius - channel_height / 2)
    triangles.extend(arc_triangles_side)
    bottom_triangles += bottom_via_triangles
    via_arc_points.append(arc_points_1)
    via_corner_points.append(via_corner_points_1)
    
    # Arc 2
    arc_points_2 = discretize_arc_2(end1, via_center, end2, direction2, numSegments, via_radius, channel_height)
    arc_triangles_side = create_arc_triangles_side(arc_points_2, channel_height, inner_arc=True) # TODO
    bottom_via_triangles, via_corner_points_2 = create_arc_triangles_xy_2(arc_points_2, via_center_points, 0.0, True, via_radius - channel_height / 2)
    triangles.extend(arc_triangles_side)
    bottom_triangles += bottom_via_triangles
    via_arc_points.append(arc_points_2)
    via_corner_points.append(via_corner_points_2)

    end1 = [luer_vertices[1][0] + (via_radius - distance_channel_circle) * direction[0], luer_vertices[1][1] - (0.5 * channel_height - via_radius) * direction[0], luer_vertices[2][2]]  
    end2 = [luer_vertices[1][0] + (via_radius * 2 - distance_channel_circle) * direction[0], luer_vertices[1][1] - 0.5 * channel_height, luer_vertices[2][2]]       
    
    direction2 = direction2 * (-1)
    # Arc 3
    arc_points_3 = discretize_arc_2(luer_vertices[1], via_center, end1, direction2, numSegments, via_radius * direction[0], channel_height)
    arc_triangles_side = create_arc_triangles_side(arc_points_3, channel_height, inner_arc=True) # TODO
    bottom_via_triangles, via_corner_points_3 = create_arc_triangles_xy_2(arc_points_3, via_center_points, 0.0, False, via_radius - channel_height / 2)
    triangles.extend(arc_triangles_side)
    bottom_triangles += bottom_via_triangles
    via_arc_points.append(arc_points_3)
    via_corner_points.append(via_corner_points_3)

    # TODO maybe add unit conversion factor here
    if via_radius != 1e-3 and channel_height * 3/4 != 1e-3 * unit_conversion_factor: # The 4th arc goes the wrong direction if the spheroid_diameter is exactly 600 um, and subsequently the via_radius is 1e-3        
        direction2 = direction2 * (-1)
    
    # Arc 4
    arc_points_4 = discretize_arc_2(end1, via_center, end2, direction2, numSegments, via_radius, channel_height)
    arc_triangles_side = create_arc_triangles_side(arc_points_4, channel_height, inner_arc=True) # TODO
    bottom_via_triangles, via_corner_points_4 = create_arc_triangles_xy_2(arc_points_4, via_center_points, 0.0, True, via_radius - channel_height / 2)
    triangles.extend(arc_triangles_side)
    bottom_triangles += bottom_via_triangles
    via_arc_points.append(arc_points_4)
    via_corner_points.append(via_corner_points_4)

    # NEW luer VERTICES IN XY PLANE
    new_luer_vertices = luer_vertices.copy()

    for i in range(4):
        new_luer_vertices[i][0] += direction[0] * via_radius / 2
        new_luer_vertices[i][2] = via_height

    if direction[0] == -1:
        direction2 = 1
    else:
        direction2 = -1 

    # 2. LUER CONNECTIONS
    # First Arc
    triangle, corner_point, arc_points = define_luer_arcs(height, new_luer_vertices[0], new_luer_vertices[1], direction, numSegments, direction2, luer_radius, luer_angle, via_radius)
    triangles.extend(triangle)
    top_corner_points.append(corner_point)
    all_arc_points.append(arc_points)
    # Second Arc
    triangle, corner_point, arc_points = define_luer_arcs(height, new_luer_vertices[1], new_luer_vertices[0], direction, numSegments, direction2, luer_radius, luer_angle, via_radius)
    triangles.extend(triangle)
    top_corner_points.append(corner_point)
    all_arc_points.append(arc_points)

    direction2 = direction2 * (-1)

    # Third Arc
    triangle, corner_point, arc_points = define_luer_arcs(height, new_luer_vertices[2], new_luer_vertices[3], direction, numSegments, direction2, luer_radius, luer_angle, via_radius)
    triangles.extend(triangle)
    top_corner_points.append(corner_point)
    all_arc_points.append(arc_points)
    # Fourth Arc
    triangle, corner_point, arc_points = define_luer_arcs(height, new_luer_vertices[3], new_luer_vertices[2], direction, numSegments, direction2, luer_radius, luer_angle, via_radius)
    triangles.extend(triangle)
    top_corner_points.append(corner_point)
    all_arc_points.append(arc_points)

    for corner in range(len(channel_corner_points)):
        for i in range(len(all_arc_points[corner])-1):
            via_triangles.append([all_arc_points[corner][i], all_arc_points[corner][i+1], channel_corner_points[corner]])

    # Define the connection between the luer and the via (this is in 2D)
    new_via_arc_points = []
    for arc in via_arc_points:
        new_arc = [(x, y, z + via_height) for x, y, z in arc]
        new_via_arc_points.append(new_arc)
    
    connection_triangles_via_luer = create_arc_triangles_xy(new_via_arc_points[1], all_arc_points[0], 0.0, True) 
    triangles.extend(connection_triangles_via_luer)
    connection_triangles_via_luer = create_arc_triangles_xy(all_arc_points[1], new_via_arc_points[3], 0.0, True) 
    triangles.extend(connection_triangles_via_luer)
    # Here the points of the luer are adapted to keep the channel in and outlet open at the top

    # 3. LUER-VIA CONNECTIONS
    if direction[0] == -1:
        # define the points based on the inlet
        channel_end_point_left = vertices[quad_list[1][3]].copy()
        channel_end_point_left[2] += channel_height
        left_channel_arc_overlap_point = channel_end_point_left.copy()
        channel_end_point_right = vertices[quad_list[1][0]].copy()
        channel_end_point_right[2] += channel_height
        right_channel_arc_overlap_point = channel_end_point_right.copy()
    else:
        channel_end_point_left = vertices[quad_list[-2 - 2 * nr_of_traps][3]].copy() # TODO this would also be affected by number of traps, once that is impl. this might be combinable with the above
        left_channel_arc_overlap_point = channel_end_point_left.copy()
        channel_end_point_right = vertices[quad_list[-2 - 2 * nr_of_traps][1]].copy() # this would also be affected by number of traps       
        right_channel_arc_overlap_point = channel_end_point_right.copy()


    nr_of_overlapping_segments = 0
    for i in range(1, len(all_arc_points[2]) + 1):
        if all_arc_points[2][-i][1] > channel_end_point_right[1]:
            nr_of_overlapping_segments += 1
        else:
            break

    actual_overlap_point = find_overlap_xy(all_arc_points[2][-nr_of_overlapping_segments], all_arc_points[2][-nr_of_overlapping_segments - 1], new_via_arc_points[0][0], channel_end_point_right)
    # distance_overlap_channel_luer = math.sqrt((luer_radius) ** 2 - (channel_height/2) ** 2) - (luer_radius)

    distance_overlap_luer_channel = actual_overlap_point[0] - channel_end_point_right[0] 

    left_channel_arc_overlap_point[0] += distance_overlap_luer_channel
    right_channel_arc_overlap_point[0] += distance_overlap_luer_channel

    right_channel_arc_overlap_points = []
    left_channel_arc_overlap_points = []

    for i in range(nr_of_overlapping_segments):
        right_channel_arc_overlap_points.append(right_channel_arc_overlap_point)
        left_channel_arc_overlap_points.append(left_channel_arc_overlap_point)
    
    connection_triangles_via_luer = create_arc_triangles_xy(new_via_arc_points[0], right_channel_arc_overlap_points + list(reversed(all_arc_points[2][:-nr_of_overlapping_segments])), 0.0, True) # right / bottom arc
    triangles.extend(connection_triangles_via_luer)
    connection_triangles_via_luer = create_arc_triangles_xy(left_channel_arc_overlap_points + list(reversed(all_arc_points[3][:-nr_of_overlapping_segments])), new_via_arc_points[2], 0.0, True) # left / top arc
    triangles.extend(connection_triangles_via_luer)

    # Add the triangles to fully close off the gap between round luer and the next closed channel TODO fix this 
    for i in range(1, nr_of_overlapping_segments):
        triangles.append([channel_end_point_left, all_arc_points[3][-(i + 1)], all_arc_points[3][-i]])
        triangles.append([channel_end_point_right, all_arc_points[2][-(i + 1)], all_arc_points[2][-i]])

    i += 1
    bottom_triangles.append([channel_end_point_left, left_channel_arc_overlap_point, all_arc_points[3][-i]])
    bottom_triangles.append([channel_end_point_right, right_channel_arc_overlap_point, all_arc_points[2][-i]])

    triangles.append([channel_end_point_left, channel_end_point_right, all_arc_points[3][-1]])
    
    triangles.extend(via_triangles)
    
    return triangles, bottom_triangles, top_corner_points, via_corner_points


def create_stl_file(nodes: list, grounds: list, channels: list, arcs: list, height: float, chip_width: float, chip_length: float, spheroid_diameter: float, svg_output_file: str, dxf_output_file: str, stl_output_file: str, mini_luer: bool, nr_of_traps: int, channel_negative: bool): #incl. bool to define positive or negative channel definition
    '''
    Create the stl file based on the given nodes, grounds, channels, arcs, and height. 
    The spheroid diameter and the number of traps are used to define the traps.
    The file name is used to save the stl file.
    The mini luer determines the use of luer vs mini luer connections for the chip definition.

    The channel_negative determines if the channel positive (e.g., for subsequent simulation) or the channel positive (e.g., for fabrication) is defined.

    The function defines the geometry based on the input, extrudes it to a 3D defintion, triangulates it and saves it as a .stl file.
    '''
    
    # PRIOR DEFINITIONS
    num_segments = 10

    unit_conversion_to_mm = True
    unit_conversion_factor = 1

    inward_triangles = []

    if unit_conversion_to_mm:
        # Convert meters to millimeters and mirror in y-direction
        unit_conversion_factor = 1e3
        for i, node in enumerate(nodes):
            nodes[i] = [node[0] * unit_conversion_factor, node[1] * unit_conversion_factor, node[2] * unit_conversion_factor]
        for i, channel in enumerate(channels):
            channels[i] = [channel[0], channel[1], channel[2] * unit_conversion_factor, channel[3] * unit_conversion_factor]
        for i, arc in enumerate(arcs):
            arcs[i] = [arc[0], [arc[1][0] * unit_conversion_factor, arc[1][1] * unit_conversion_factor], arc[2], arc[3] * unit_conversion_factor, arc[4] * unit_conversion_factor, arc[5]]

        height *= unit_conversion_factor

    chip_width = chip_width * unit_conversion_factor
    chip_length = chip_length * unit_conversion_factor

    channels_per_node = define_channels_per_node(nodes, channels)
    quads = define_quads_at_nodes(nodes, channels_per_node)

    channel_height = height

    vertices, quad_list = define_vertices_and_quad_list(nodes, quads, channel_height)
    channels_per_node = define_channels_per_node(nodes, channels)

    # QUAD FACE DEFINITION
    quad_faces = define_quad_faces_xy(quad_list) # top and bottom faces
    len_quad_faces = len(quad_faces)
    quad_faces_bottom = quad_faces[:len_quad_faces//2]
    quad_faces_top = quad_faces[len_quad_faces//2:]
    quad_faces_side = define_faces_side(quad_faces_bottom, quad_faces_top)

    # CHANNEL FACE DEFINITION
    # channel_faces_bottom = define_channel_faces_xy(nodes, channels, quad_list) # defines the channel faces on the bottom
    channel_faces_bottom = define_channel_faces_xy(nodes, channels, quad_faces_bottom)
    channel_faces_top = define_channel_faces_xy(nodes, channels, quad_faces_top)
    channel_faces_side = define_faces_side(channel_faces_bottom, channel_faces_top)

    # # remove the top face of the trap inflow channel
    # for i in range(nr_of_traps):
    #     channel_faces_top = channel_faces_top[:-1 + i * 2] + channel_faces_top[1 + i * 2:]

    trap_out_bottom_faces = []
    trap_out_top_faces = []

    if channel_negative:
        # Remove the trap channel faces
        channel_faces_top_adapted = channel_faces_top[:-(2 * nr_of_traps)]
        channel_faces_top_adapted = channel_faces_top_adapted[1:-1]
        for i in range(1, nr_of_traps + 1):
            channel_faces_top_adapted.append(channel_faces_top[-(2 * i) + 1])
        channel_faces_bottom_adapted = [] # The chip definition is open on the bottom
        quad_faces_bottom = []
    else: # remove the bottom face of the trap inflow channel
        channel_faces_bottom_adapted = channel_faces_bottom[:-(2 * nr_of_traps)]
        channel_faces_top_adapted = channel_faces_top[:-(2 * nr_of_traps)]
        for i in range(nr_of_traps):
            trap_out_bottom_faces.append(channel_faces_bottom[-(2 * nr_of_traps) + i * 2 + 1])
            trap_out_top_faces.append(channel_faces_top[-(2 * nr_of_traps) + i * 2 + 1])
        channel_faces_bottom_adapted.extend(trap_out_bottom_faces)
        channel_faces_top_adapted.extend(trap_out_top_faces)
        

    # all_faces = quad_faces_bottom + quad_faces_top + channel_faces_bottom_adapted + channel_faces_top_adapted + quad_faces_side + channel_faces_side
    all_faces = quad_faces_bottom + quad_faces_top + channel_faces_bottom_adapted + channel_faces_top_adapted + quad_faces_side + channel_faces_side
    top_faces = quad_faces_top + channel_faces_top_adapted # these are visible from the bottom side as the channel is open

    # REMOVE DUPLICATE FACES
    non_overlapping_faces = remove_shared_faces(all_faces)

    # # PLOT VERTICES
    # plot_vertices(vertices)

    # TRIANGULATE
    channel_triangles = triangulation(non_overlapping_faces, vertices, xy_orientation=True)
    channel_top_triangles = triangulation(top_faces, vertices, xy_orientation=True)
    if len(arcs) != 0:
        arc_triangles, bottom_arc_triangles, corner_points_arc = define_arcs(nodes, quad_list, vertices, arcs, num_segments, channel_height, nr_of_traps, channel_negative)
        triangles = channel_triangles # + arc_triangles
    else:
        triangles = channel_triangles

    # TRAP DEFINITION
    trap_triangles, trap_triangles_bottom, corner_points_trap, trap_start_and_end_nodes = trap_geometry(nodes, quad_list, vertices, channels, channel_negative, num_segments, nr_of_traps)
    #chip_triangles += trap_triangles

    # GROUND DEFINITON 
    ground_vertices = []
    for nodeId in grounds:
        ground_vertice_0 = vertices[quad_list[nodeId][0]]
        ground_vertice_1 = vertices[quad_list[nodeId][2]] # changed from 2

        # add z direction to the bottom vertices
        ground_vertice_0_top = ground_vertice_0.copy()
        ground_vertice_0_top[2] += channel_height
        ground_vertice_1_top = ground_vertice_1.copy()
        ground_vertice_1_top[2] += channel_height

        ground_vertices.append([ground_vertice_0, ground_vertice_1, ground_vertice_0_top, ground_vertice_1_top])
    
    ground_vertices = np.array(ground_vertices, dtype=np.float64)

    # create_svg_network_1D(nodes, channels, filename='../tests/network1D.svg')
    create_svg_network_2D(vertices, nodes, quad_list, channel_faces_top, arcs, trap_start_and_end_nodes, svg_output_file) # the trap is missing TODO

    # save_to_dxf(nodes, channels, arcs, trap_start_and_end_nodes, filename="network.dxf")
    export_dxf(vertices, nodes, quad_list, channel_faces_top,  arcs, trap_start_and_end_nodes, dxf_output_file)

    # CHANNEL STRUCTURE (negative of positive for fabricaiton or simulation respectively)
    if channel_negative: # the result should be the negative of the channel embedded in a chip, ready to print
        chip_thickness = 3e-3 * unit_conversion_factor
        fase_side_length = 1e-3 * unit_conversion_factor

        # Define the size of the luer connections
        if mini_luer:
            luer_radius = 1.32e-3 * unit_conversion_factor
            luer_angle = 91.94
        else:
            luer_radius =  1.95e-3 * unit_conversion_factor
            luer_angle = 91.72

        # Define the chip block
        chip_triangles, bottom_chip_triangles, chip_top_corners, chip_bottom_corners = add_chip_triangles(chip_thickness, chip_width, chip_length, fase_side_length, unit_conversion_factor)

        # Add luer connections, here the grounds are the luer connections
        luer_direction = [-1, 0, 0]
        luer_triangles_connection_inlet, via_bottom_triangles_1, luer_corner_points_inlet, via_corner_points_inlet = add_luer_connections(ground_vertices[0], vertices, luer_direction, num_segments, chip_thickness, luer_radius, luer_angle, channel_height, quad_list, nr_of_traps, unit_conversion_factor)
        luer_direction = [1, 0, 0]
        luer_triangles_connection_outlet, via_bottom_triangles_2, luer_corner_points_outlet, via_corner_points_outlet = add_luer_connections(ground_vertices[1], vertices, luer_direction, num_segments, chip_thickness, luer_radius, luer_angle, channel_height, quad_list, nr_of_traps, unit_conversion_factor)
        luer_triangles_connection = luer_triangles_connection_inlet + luer_triangles_connection_outlet
        luer_corner_points = luer_corner_points_inlet + luer_corner_points_outlet
        corner_points_via = via_corner_points_inlet + via_corner_points_outlet
        triangles += luer_triangles_connection

        # Define the top layer of the chip
        triangles += add_chip_top_layer_triangles(chip_top_corners, luer_corner_points)

        # Define the bottom layer of the chip
        channel_width = 4/3 * spheroid_diameter * unit_conversion_factor # TODO maybe make this better dependent on the actual channel_width
        bottom_chip_triangles += add_chip_bottom_layer_triangles(chip_bottom_corners, corner_points_via, list(reversed(corner_points_trap)), corner_points_arc, channel_width, nr_of_traps) # corner point arc is outer point, inner point left to right, trap is left then right point

        triangles += chip_triangles
        inward_triangles = bottom_chip_triangles + channel_top_triangles  + bottom_arc_triangles + trap_triangles + via_bottom_triangles_1 + via_bottom_triangles_2 

    elif channel_negative == False: # the result should be the positive of the channel, the inlets and outlets defined as faces to be able to load the channel network into a simulation tool 
        ground_triangles = []
        
        for i in ground_vertices:
            ground_triangles.append([i[0], i[3], i[1]])
            ground_triangles.append([i[0], i[2],i[3]])

        triangles += ground_triangles
            

    # Correct the direction of all triangles
    if channel_negative:
        # inward_triangles = bottom_chip_triangles + channel_top_triangles + bottom_arc_triangles + trap_triangles + via_bottom_triangles_1 + via_bottom_triangles_2
        triangles += arc_triangles
        triangles = correct_triangle_winding(triangles, outward=True)
        inward_triangles = correct_triangle_winding(inward_triangles, outward=False)
        triangles += inward_triangles
    else:
        triangles += trap_triangles + arc_triangles + bottom_arc_triangles + trap_triangles_bottom
        triangles = correct_triangle_winding(triangles, outward=True)

    len_triangles = len(triangles)
    triangle_counter = 0

    # Create the mesh with the total number of triangles
    mesh_data = mesh.Mesh(np.zeros(len_triangles, dtype=mesh.Mesh.dtype))

    for triangle in triangles:
        mesh_data.vectors[triangle_counter] = np.array(triangle, dtype=np.float64)
        triangle_counter += 1
    
    mesh_data.save(stl_output_file)
    # To transform the stl file to ascii type ```stl2ascii name.stl new_name.stl in your terminal

    # you might be able to automatically update the normals
    print(f'\nSTL file "{stl_output_file}" created successfully.\n')

    # generate_inlet_and_outlet(ground_vertices)

def correct_triangle_winding(triangles, outward=True):
    """
    Ensures all triangles have a consistent winding order.
    
    :param triangles: List of triangles, each defined by three vertices [(x, y, z), ...].
    :param outward: Whether the triangles should face outward (default: True).
    :return: List of triangles with consistent winding order.
    """
    corrected_triangles = []
    for triangle in triangles:
        v0, v1, v2 = np.array(triangle[0]), np.array(triangle[1]), np.array(triangle[2])
        # Calculate the normal
        normal = np.cross(v1 - v0, v2 - v0)
        # Check orientation
        if outward:
            if np.dot(normal, [0, 0, 1]) < 0:  # Adjust this normal check for desired outward direction
                corrected_triangles.append([triangle[0], triangle[2], triangle[1]])
            else:
                corrected_triangles.append(triangle)
        else:
            if np.dot(normal, [0, 0, 1]) > 0:  # Adjust this normal check for desired inward direction
                corrected_triangles.append([triangle[0], triangle[2], triangle[1]])
            else:
                corrected_triangles.append(triangle)
    return corrected_triangles

def plot_nodes(nodes: list, channels: list):
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

    # Plot channels as lines between nodes
    # for start_node, end_node, width in channels:
    for start_node, end_node, width, height in channels:
        start_x, start_y = nodes[start_node][:2]
        end_x, end_y = nodes[end_node][:2]
        plt.plot([start_x, end_x], [start_y, end_y], 'k-', lw=max(width * 0.1, 1), label='Channels' if start_node == channels[0][0] else "")

    plt.title('Node Positions in XY Plane')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_vertices(vertices: list):
    '''
    Plots the vertices of the network.
    Can be used to verify the vertice position and numbering.
    '''

    vertices = vertices[:len(vertices)//2]

    x_values = [vertex[0] for vertex in vertices]
    y_values = [vertex[1] for vertex in vertices]

    plt.scatter(x_values, y_values)

    for i, vertex in enumerate(vertices):
        plt.text(vertex[0], vertex[1], str(i), fontsize=8, ha='left', va='bottom')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Vertices Plot')
    plt.show()

def plot_vertices_and_arcs(nodes, arcs):
    x_values = [node[0] for node in nodes]
    y_values = [node[1] for node in nodes]
    
    plt.scatter(x_values, y_values, color='blue', label='Nodes')
    
    for arc in arcs:
        start = nodes[arc[0]]
        center = arc[1]
        end = nodes[arc[2]]
        
        # Plot start, center, and end points
        plt.scatter(start[0], start[1], color='green', label='Start' if arc == arcs[0] else "")
        plt.scatter(center[0], center[1], color='red', label='Center' if arc == arcs[0] else "")
        plt.scatter(end[0], end[1], color='purple', label='End' if arc == arcs[0] else "")
        
        # Plot the arc
        arc_points = discretize_arc(start, center, end, arc[5], 100, 0, 0)
        arc_x = [point[0] for point in arc_points]
        arc_y = [point[1] for point in arc_points]
        plt.plot(arc_x, arc_y, color='black', linestyle='dotted', label='Arc' if arc == arcs[0] else "")

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Nodes and Arcs Plot')
    plt.legend()
    plt.show()

def create_svg_network_2D(vertices: list, nodes: list, quad_list: list, channel_faces: list, arcs: list, trap_start_and_end_nodes: list, filename):
    header = '''<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="50" height="25" viewBox="-10 0 60 10">\n'''
    footer = '</svg>'
    
    with open(filename, 'w') as svg_file:
        svg_file.write(header)
        
        # CHANNELS
        for channel in channel_faces:
            coords = [vertices[i] for i in channel]
            points_str = " ".join([f"{pt[0]},{pt[1]}" for pt in coords])
            # Write the polygon to the SVG
            svg_file.write(f'<polygon points="{points_str}" style="fill:none;stroke:black;stroke-width:0.1" />\n')
        
        # QUADS
        for quad in quad_list:
            quad_coords = [vertices[i] for i in quad]
            points_str = " ".join([f"{pt[0]},{pt[1]}" for pt in quad_coords])
            # Write the polygon to the SVG
            svg_file.write(f'<polygon points="{points_str}" style="fill:none;stroke:black;stroke-width:0.1" />\n')

        # Write arcs
        for arc in arcs:
            arc[1] = arc[1] + [nodes[arc[0]][2]]

            quad_1 = quad_list[arc[0]]
            quad_2 = quad_list[arc[2]]

            if nodes[arc[0]][0] < nodes[arc[2]][0]: # arc goes in + x direction # TODO combine this into a smaller loop
                if nodes[arc[0]][1] < nodes[arc[2]][1]: # arc goes in + y direction
                    if arc[1][1] == nodes[arc[0]][1]: # arc center is in +x direction
                        end1 = quad_1[3] # 2. try
                        end2 = quad_1[2]
                        start1 = quad_2[3]
                        start2 = quad_2[0]
                    elif arc[1][0] == nodes[arc[0]][0]: # arc center is in +y direction
                        start1 = quad_1[1]
                        start2 = quad_1[2]
                        end1 = quad_2[1]
                        end2 = quad_2[0]
                    else:
                        print("Error: arc1a", arc) 
                elif nodes[arc[0]][1] > nodes[arc[2]][1]: # arc goes in - y direction
                    if arc[1][1] == nodes[arc[0]][1]:
                        start1 = quad_1[0]
                        start2 = quad_1[1]
                        end1 = quad_2[0]
                        end2 = quad_2[3]
                    elif arc[1][0] == nodes[arc[0]][0]: # arc center is in -y direction
                        end1 = quad_1[2] # 2. try
                        end2 = quad_1[1]
                        start1 = quad_2[2]
                        start2 = quad_2[3]
                    else:
                        print("Error: arc1b", arc) 
                else:
                    print("Error: arc2", arc) 
            elif nodes[arc[0]][0] > nodes[arc[2]][0]: # arc goes in -x direction
                if nodes[arc[0]][1] < nodes[arc[2]][1]: # arc goes in + y direction
                    if arc[1][1] == nodes[arc[0]][1]: # arc center is in -x direction
                        end1 = quad_1[3] # 2. try
                        end2 = quad_1[2]
                        start1 = quad_2[1]
                        start2 = quad_2[2]
                    elif arc[1][0] == nodes[arc[0]][0]: # arc center is in +y direction
                        start1 = quad_1[0]
                        start2 = quad_1[3]
                        end1 = quad_2[0]
                        end2 = quad_2[1]
                    else:
                        print("Error: arc1", arc) 
                elif nodes[arc[0]][1] > nodes[arc[2]][1]: # arc goes in - y direction
                    if arc[1][1] == nodes[arc[0]][1]: # arc center is in -x direction
                        end1 = quad_1[1] # 2. try
                        end2 = quad_1[0]
                        start1 = quad_2[1]
                        start2 = quad_2[2]
                    elif arc[1][0] == nodes[arc[0]][0]: # arc center is in -y direction
                        end1 = quad_1[3] # 2. try
                        end2 = quad_1[0]
                        start1 = quad_2[3]
                        start2 = quad_2[2]
                    else:
                        print("Error: arc1", arc)    
                else:
                    print("Error: arc2", arc) 
            else:
                print("Error: arc3", arc)

            radius = vertices[start1][0] - vertices[end1][0]
            svg_file.write(f'<path d="M{vertices[start1][0]},{vertices[start1][1]} A{radius},{radius} 0 0,1 {vertices[end1][0]},{vertices[end1][1]}" fill="none" stroke="blue" stroke-width="0.1" />\n')
            radius = vertices[start2][0] - vertices[end2][0]
            svg_file.write(f'<path d="M{vertices[start2][0]},{vertices[start2][1]} A{radius},{radius} 0 0,1 {vertices[end2][0]},{vertices[end2][1]}" fill="none" stroke="blue" stroke-width="0.1" />\n')

        # Define the trap arcs
        for trap in range(len(trap_start_and_end_nodes)): # list containing lists with 2 start and 2 end nodes per trap
            radius = -(trap_start_and_end_nodes[trap][0][1] - trap_start_and_end_nodes[trap][2][1]) / 2
            svg_file.write(f'<path d="M{trap_start_and_end_nodes[trap][0][0]},{trap_start_and_end_nodes[trap][0][1]} A{radius},{radius} 0 0,0 {trap_start_and_end_nodes[trap][1][0]},{trap_start_and_end_nodes[trap][1][1]}" fill="none" stroke="blue" stroke-width="0.1" />\n')
            svg_file.write(f'<path d="M{trap_start_and_end_nodes[trap][2][0]},{trap_start_and_end_nodes[trap][2][1]} A{radius},{radius} 0 0,1 {trap_start_and_end_nodes[trap][3][0]},{trap_start_and_end_nodes[trap][3][1]}" fill="none" stroke="blue" stroke-width="0.1" />\n')
        
        svg_file.write(footer)

    print(f'\nSVG file "{filename}" created successfully.')

def save_to_dxf(nodes, channels, arcs, trap_start_and_end_nodes, filename="network.dxf"): 
    """
    Save the channel network as a 2D DXF file. Here the channels are represented by black lines, i.e., the channel width is not considered.
    The trap is also not closed as the arcs closing the trap are dependent on its width.
    """
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Add channels as polylines
    for channel in channels:
        start_node = nodes[channel[0]]
        end_node = nodes[channel[1]]
        msp.add_line(start=(start_node[0], start_node[1]), end=(end_node[0], end_node[1]))

    # Add arcs
    for arc in arcs:
        direction = True
        if nodes[arc[0]][0] < nodes[arc[2]][0]:
            if nodes[arc[0]][1] < nodes[arc[2]][1] and arc[1][0] == nodes[arc[0]][0]:
                direction = False
            elif nodes[arc[0]][1] > nodes[arc[2]][1] and arc[1][1] == nodes[arc[0]][1]:
                direction = False
        elif nodes[arc[0]][0] > nodes[arc[2]][0]:
            if arc[1][0] == nodes[arc[0]][0]:
                direction = False
        start = nodes[arc[0]]
        center = arc[1]  # Arc center is stored directly
        end = nodes[arc[2]]
        radius = ((start[0] - center[0])**2 + (start[1] - center[1])**2) ** 0.5
        angle_start = math.degrees(math.atan2(start[1] - center[1], start[0] - center[0]))
        angle_end = math.degrees(math.atan2(end[1] - center[1], end[0] - center[0]))
        print(f"angle_start: {angle_start}, angle_end: {angle_end}, direction: {direction}")

        if direction:
            msp.add_arc(center=(center[0], center[1]), radius=radius, start_angle=angle_end, end_angle=angle_start)        
        else:
            msp.add_arc(center=(center[0], center[1]), radius=radius, start_angle=angle_start, end_angle=angle_end)

    # Save the DXF file
    doc.saveas(filename)
    print(f"DXF file saved as {filename}")

def export_dxf(vertices, nodes, quad_list, channel_faces_xy, arcs, trap_start_and_end_nodes, filename):
    '''
    Saves the channel network as a dxf file. Represents the 2D network, channels are represented as rectangles and arcs are two rounded arcs. 
    The result looks very similar to the generated SVG file.
    '''

    # Create a new DXF document
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Iterate over channel faces (which already include thickness)
    for face in channel_faces_xy:
        # Get the actual (x, y) coordinates for each vertex in the face
        pts = [vertices[i][:2] for i in face]  # Extract only (x, y) from (x, y, z)

        # Add the polyline to the DXF
        msp.add_lwpolyline(pts, close=True, dxfattribs={"color": 7, "lineweight": 1})

    # Draw quad outlines in DXF
    for quad in quad_list:
        pts = [vertices[i][:2] for i in quad]  # Extract only (x, y)
        msp.add_lwpolyline(pts, close=True, dxfattribs={"color": 7, "lineweight": 1})

    for arc in arcs:
        arc[1] = arc[1] + [nodes[arc[0]][2]]
        quad_1 = quad_list[arc[0]]
        quad_2 = quad_list[arc[2]]

        direction = True

        if nodes[arc[0]][0] < nodes[arc[2]][0]: # arc goes in + x direction 
            if nodes[arc[0]][1] < nodes[arc[2]][1]: # arc goes in + y direction
                if arc[1][1] == nodes[arc[0]][1]: # arc center is in +x direction
                    start1 = quad_1[3]
                    start2 = quad_1[2]
                    end1 = quad_2[3]
                    end2 = quad_2[0]
                elif arc[1][0] == nodes[arc[0]][0]: # arc center is in +y direction
                    direction = False
                    start1 = quad_1[1]
                    start2 = quad_1[2]
                    end1 = quad_2[1]
                    end2 = quad_2[0]
                else:
                    print("Error: arc1", arc) 
            elif nodes[arc[0]][1] > nodes[arc[2]][1]: # arc goes in - y direction
                if arc[1][1] == nodes[arc[0]][1]:
                    direction = False
                    start1 = quad_1[0]
                    start2 = quad_1[1]
                    end1 = quad_2[0]
                    end2 = quad_2[3]
                elif arc[1][0] == nodes[arc[0]][0]: # arc center is in -y direction
                    start1 = quad_1[2]
                    start2 = quad_1[1]
                    end1 = quad_2[2]
                    end2 = quad_2[3]
                else:
                    print("Error: arc1", arc) 
            else:
                print("Error: arc2", arc) 
        elif nodes[arc[0]][0] > nodes[arc[2]][0]: # arc goes in -x direction
            if nodes[arc[0]][1] < nodes[arc[2]][1]: # arc goes in + y direction
                if arc[1][1] == nodes[arc[0]][1]: # arc center is in -x direction
                    start1 = quad_1[3]
                    start2 = quad_1[2]
                    end1 = quad_2[1]
                    end2 = quad_2[2]
                elif arc[1][0] == nodes[arc[0]][0]: # arc center is in +y direction
                    direction = False
                    start1 = quad_1[0]
                    start2 = quad_1[3]
                    end1 = quad_2[0]
                    end2 = quad_2[1]
                else:
                    print("Error: arc1", arc) 
            elif nodes[arc[0]][1] > nodes[arc[2]][1]: # arc goes in - y direction
                if arc[1][1] == nodes[arc[0]][1]: # arc center is in -x direction
                    start1 = quad_1[1]
                    start2 = quad_1[0]
                    end1 = quad_2[1]
                    end2 = quad_2[2]
                elif arc[1][0] == nodes[arc[0]][0]: # arc center is in -y direction
                    start1 = quad_1[3]
                    start2 = quad_1[0]
                    end1 = quad_2[3]
                    end2 = quad_2[2]

        center = arc[1]
        radius1 = ((vertices[start1][0] - center[0])**2 + (vertices[start1][1] - center[1])**2) ** 0.5
        radius2 = ((vertices[start2][0] - center[0])**2 + (vertices[start2][1] - center[1])**2) ** 0.5

        angle_start1 = math.degrees(math.atan2(vertices[start1][1] - center[1], vertices[start1][0] - center[0]))
        angle_end1 = math.degrees(math.atan2(vertices[end1][1] - center[1], vertices[end1][0] - center[0]))
        angle_start2 = math.degrees(math.atan2(vertices[start2][1] - center[1], vertices[start2][0] - center[0]))
        angle_end2 = math.degrees(math.atan2(vertices[end2][1] - center[1], vertices[end2][0] - center[0]))

        if direction:
            msp.add_arc(center=(center[0], center[1]), radius=radius1, start_angle=angle_end1, end_angle=angle_start1, dxfattribs={"lineweight": 1})
            msp.add_arc(center=(center[0], center[1]), radius=radius2, start_angle=angle_end2, end_angle=angle_start2, dxfattribs={"lineweight": 1})
        else:
            msp.add_arc(center=(center[0], center[1]), radius=radius1, start_angle=angle_start1, end_angle=angle_end1, dxfattribs={"lineweight": 1})
            msp.add_arc(center=(center[0], center[1]), radius=radius2, start_angle=angle_start2, end_angle=angle_end2, dxfattribs={"lineweight": 1})


    for trap in range(len(trap_start_and_end_nodes)):
        start1 = trap_start_and_end_nodes[trap][0]  # (x, y) coordinate
        end1 = trap_start_and_end_nodes[trap][1]
        start2 = trap_start_and_end_nodes[trap][2]
        end2 = trap_start_and_end_nodes[trap][3]

        # Compute radius as half the distance between start1 and start2
        dx = start1[0] - start2[0]
        dy = start1[1] - start2[1]
        radius = math.sqrt(dx**2 + dy**2) / 2

        # Compute center of arc
        center_x = (start1[0] + start2[0]) / 2
        center_y = (start1[1] + start2[1]) / 2

        # Compute start and end angles in degrees
        end_angle_1 = math.degrees(math.atan2(start1[1] - center_y, start1[0] - center_x))
        start_angle_1 = math.degrees(math.atan2(end1[1] - center_y, end1[0] - center_x))

        start_angle_2 = math.degrees(math.atan2(start2[1] - center_y, start2[0] - center_x))
        end_angle_2 = math.degrees(math.atan2(end2[1] - center_y, end2[0] - center_x))

        # Draw arcs (assuming dxf is being created with ezdxf)
        msp.add_arc(
            center=(center_x, center_y),
            radius=radius,
            start_angle=start_angle_1,
            end_angle=end_angle_1,
            dxfattribs={"lineweight": 1}
        )
        msp.add_arc(
            center=(center_x, center_y),
            radius=radius,
            start_angle=start_angle_2,
            end_angle=end_angle_2,
            dxfattribs={"lineweight": 1}
        )

    # Save the DXF file
    doc.saveas(filename)
    print(f"DXF file saved: {filename}")


def generate_geometry(filename, spheroid_diameter, svg_output_file, dxf_output_file, stl_output_file, mini_luer, nr_of_traps, channel_negative: bool):
    '''
    Reads in the JSON definiton of the channel network and generates the resulting STL file.
    '''
    nodes, grounds, channels, arcs, height, chip_width, chip_length = read_in_network_file(filename)
    create_stl_file(nodes, grounds, channels, arcs, height, chip_width, chip_length, spheroid_diameter, svg_output_file, dxf_output_file, stl_output_file, mini_luer, nr_of_traps, channel_negative)

if __name__ == "__main__":
    svg_output_file = 'output.svg'
    dxf_output_file = 'output.dxf'
    stl_output_file = 'output.stl'
    filename = '/Users/maria/Documents/GitHub/spheroid-trap-designer/spheroid_trap_network.json'
    nodes, grounds, channels, arcs, height, chip_width, chip_length = read_in_network_file(filename)

    spheroid_diameter = 400e-6 
    mini_luer = False
    nr_of_traps = 1

    plot_nodes(nodes, channels)

    create_stl_file(nodes, grounds, channels, arcs, height, chip_width, chip_length, spheroid_diameter, svg_output_file, dxf_output_file, stl_output_file, mini_luer, nr_of_traps, True)

