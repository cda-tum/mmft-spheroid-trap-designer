# **Spheroid Trap Designer – OpenFOAM Simulation Setup**

The **Spheroid Trap Designer** allows you to generate a customized geometry based on your specified parameters. Once the geometry is created, you can simulate fluid flow within it using **OpenFOAM**, an open-source computational fluid dynamics (CFD) tool.

Before proceeding, ensure that **OpenFOAM** is installed on your system. Installation instructions can be found on the [official OpenFOAM website](https://openfoam.org/download/). The set up described here is tailored for OpenFOAM 12.

## **Simulation Setup Guide**

### **1. Generate the Geometry**
- Define your parameters and generate the spheroid trap geometry via the **web interface** or by running the provided command-line tool.
- The generated **STL file** will represent the 3D shape of the trap. The generated STL is defined in mm, here the STL is automatically scaled to size.

### **2. Prepare for Simulation**
- Copy the generated **STL file** into the following directory: (You might need to generate the ```geometry``` folder)
  ```sh
  /simulation/openFoam/constant/geometry
  ```
- Rename the STL file to ``` trap.stl ```
- Modify the **vertices** in the **blockMeshDict** file to ensure the mesh properly fits around your geometry:  
  ```sh
  /simulation/openFoam/system/blockMeshDict
  ```
  **Important:** The block mesh needs to "cut-off" the inlet and the outlet so these can be defined as specialized boundaries, i.e., that an inflow or outflow can be defined. Currently the block mesh is defined for a trap input for spheroids with a diameter of 300 µm.

### **3. Run the Fluid Flow Simulation**
- Navigate to the **OpenFOAM simulation directory**:  
  ```sh
  cd /simulation/openFoam
  ```
- Start the simulation by executing:  
  ```sh
  run
  ```

This will generate the computational mesh and simulate fluid flow within your custom spheroid trap.
