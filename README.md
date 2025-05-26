# Spheroid-Trap-Designer

<p align="center">
<img src="https://www.cda.cit.tum.de/research/microfluidics/logo-microfluidics-toolkit.png" style="margin:auto;width:60%"/>
</p>

The MMFT spheroid-trap designer is a tool that generates a microfluidic chip and/or channel design for trapping spheroids based on hydrodynamic principles in a dedicated location on a chip, by employing a SAT solver to optimize the design process and automatically compute the required parameters while adhering to ISO guidelines. It is possible to chain several traps together and thereby trap several spheroids. These traps are especially useful for studying complex biological processes such as the vascularization of spheroids and organoids [1].

The tool is available online: [https://www.cda.cit.tum.de/mmft-spheroid-trap-desiger](https://www.cda.cit.tum.de/mmft-spheroid-trap-designer/).
It is part of the [Munich Microfluidics Toolkit (MMFT)](https://www.cda.cit.tum.de/research/microfluidics/munich-microfluidics-toolkit/) by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the Technical University of Munich.

If you have any questions about the fabrication and the trapping experiments feel free to contact us at microfab@cea.fr.
If you have questions about using or the development of the code write us via microfluidics.cda@xcit.tum.de or by creating an issue on GitHub.

## System Requirements
The project is tested with python 3.11. To install the required python packages please type the following in your command line:
```bash
    pip install -r requirements.txt
```

## Run
To run the script, via the command line, type: 
```bash
    python spheroid_all.py
```

You can adapt the parameters in the same script.

## Simulation Setup

The **Spheroid Trap Designer** allows you to generate a customized geometry based on your specified parameters. Once the geometry is created, you can simulate fluid flow within it. For this you can use **OpenFOAM**, an open-source computational fluid dynamics (CFD) tool, COMSOL, a commercial CFD tool, or any other tool. For OpenFOAM and COMSOL we have included some of the simulation files that can be adapted for personal use. 

### OpenFOAM

Before proceeding, ensure that **OpenFOAM** is installed on your system. Installation instructions can be found on the [official OpenFOAM website](https://openfoam.org/download/).

### **Simulation Setup Guide**

#### **1. Generate the Geometry**
- Define your parameters and generate the spheroid trap geometry via the **web interface** or by running the provided command-line tool.
- The generated **STL file** will represent the 3D shape of the trap.

#### **2. Prepare for Simulation**
- Copy the generated **STL file** into the following directory:  
  ```sh
  /simulation/openFoam/constant/geometry
  ```
- Modify the **vertices** in the **blockMeshDict** file to ensure the mesh properly fits around your geometry:  
  ```sh
  /simulation/openFoam/system/blockMeshDict
  ```

#### **3. Run the Fluid Flow Simulation**
- Navigate to the **OpenFOAM simulation directory**:  
  ```sh
  cd /simulation/openFoam
  ```
- Start the simulation by executing:  
  ```sh
  run
  ```

This will generate the computational mesh and simulate fluid flow within your custom spheroid trap.

### COMSOL

The comsol simulation file will be uploaded in the simulation folder. The simulation parameters can be adapted via the user interface of the tool.

## References

[[1]](https://doi.org/10.1038/s41467-024-45710-4) Quintard, C., Tubbs, E., Jonsson, G. et al. A microfluidic platform integrating functional vascularized organoids-on-chip. Nat Commun 15, 1452 (2024).

**Developed in collaboration between:**


<p align="center">
  <picture>
    <img src="interface/tum_x_ceaLeti.png" alt="TUM × CEA-Leti" width="60%">
  </picture>
</p>
