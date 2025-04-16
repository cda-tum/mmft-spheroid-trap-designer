import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING
import uuid

from flask import Flask, cli, request, jsonify, send_file, render_template,  url_for, send_from_directory
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import spheroid_all

# if TYPE_CHECKING or sys.version_info < (3, 10, 0):  # pragma: no cover
#     import importlib_resources as resources
# else:
#     from importlib import resources

app = Flask(
    __name__,
    template_folder="interface",  # Template folder for HTML
    static_folder="interface/static"  # Static folder for CSS, JS, images
)
PREFIX = "/mmft-spheroid-trap-designer/"

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to handle the computation
@app.route('/compute', methods=['POST'])
def compute():
    data = request.json
    spheroid_diameter = data.get('spheroid_diameter')
    nr_of_traps = data.get('nr_of_traps')
    channel_negative = data.get('channel_negative', False)

    if spheroid_diameter is None:
        return jsonify({'error': 'No integer input provided'}), 400

    # Ensure integer_input is an float
    try:
        spheroid_diameter = float(spheroid_diameter)
    except ValueError:
        return jsonify({'error': 'Invalid spheroid diameter input'}), 400

    spheroid_diameter = spheroid_diameter * 1e-6

    # Ensure no of traps is an integer
    try:
        nr_of_traps = int(nr_of_traps)
    except ValueError:
        return jsonify({'error': 'Invalid no. of traps input'}), 400
    
    output_id = str(uuid.uuid4())
    output_dir = os.path.join(app.static_folder, 'outputs', output_id)
    os.makedirs(output_dir, exist_ok=True)

    svg_path = os.path.join(output_dir, 'output2D.svg')
    dxf_path = os.path.join(output_dir, 'output2D.dxf')
    stl_path = os.path.join(output_dir, 'output3D.stl')
    json_path = os.path.join(output_dir, 'gui_test.json')

    Q_ab = spheroid_all.run(spheroid_diameter, mini_luer=True, nr_of_traps=nr_of_traps, channel_negative=channel_negative, svg_output_file=svg_path, dxf_output_file=dxf_path, stl_output_file=stl_path, filename=json_path)

    # Confirm the SVG file was created
    # svg_path = os.path.join(app.static_folder, 'output2D.svg')
    if not os.path.exists(svg_path):
        return jsonify({'error': 'SVG file not created'}), 500
    
    # Confirm the DXF file was created
    # dxf_path = os.path.join(app.static_folder, 'output2D.dxf')
    if not os.path.exists(dxf_path):
        return jsonify({'error': 'DXF file not created'}), 500
    
    # Confirm the STL file was created
    # stl_path = os.path.join(app.static_folder, 'output3D.stl')
    if not os.path.exists(stl_path):
        return jsonify({'error': 'STL file not created'}), 500
    
    # Send JSON response with data and path to the SVG
    return jsonify({'result_data': 'Calculation complete.<br>Calculated flow ratio Q<sub>ab</sub>= {:.3f}'.format(Q_ab), 
                    'output_id': output_id,
                    'svg_filename': 'output2D.svg', 
                    'dxf_filename': 'output2D.dxf',
                    'stl_filename': 'output3D.stl'})

# Serve the SVG file
@app.route('/svg/<output_id>/output2D.svg')
def get_svg(output_id):
    file_path = os.path.join(app.static_folder, 'outputs', output_id, 'output2D.svg')
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, mimetype='image/svg+xml')

@app.route('/dxf/<output_id>/output2D.dxf')
def get_dxf(output_id):
    file_path = os.path.join(app.static_folder, 'outputs', output_id, 'output2D.dxf')
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, mimetype='application/dxf')

@app.route('/stl/<output_id>/output3D.stl')
def get_stl(output_id):
    file_path = os.path.join(app.static_folder, 'outputs', output_id, 'output3D.stl')
    print(f"Serving STL file: {file_path}")
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, mimetype='application/sla')

def start_server(
    skip_question: bool = False,
    activate_logging: bool = False,
    target_location: str | None = None,
    debug_flag: bool = False,
) -> None:
    """Start the server."""
    # if not target_location:
    #     target_location = str(resources.files("mmft.spheroid-trap-designer") / "static" / "files")

    # Server(
    #     target_location=target_location,
    #     skip_question=skip_question,
    #     activate_logging=activate_logging,
    # )
    print(
        "Server is hosted at: http://127.0.0.1:5000" + PREFIX + ".",
        "To stop it, interrupt the process (e.g., via CTRL+C). \n",
    )

    # This line avoid the startup-message from flask
    cli.show_server_banner = lambda *_args: None

    # if not activate_logging:
    #     log = logging.getLogger("werkzeug")
    #     log.disabled = True

    app.run(debug=debug_flag)


if __name__ == "__main__":
    start_server(debug_flag=True)
