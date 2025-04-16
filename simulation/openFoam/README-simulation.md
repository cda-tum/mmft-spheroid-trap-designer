# **Spheroid Trap Designer – OpenFOAM Simulation Setup**

The **Spheroid Trap Designer** allows you to generate a customized geometry based on your specified parameters. Once the geometry is created, you can simulate fluid flow within it using **OpenFOAM**, an open-source computational fluid dynamics (CFD) tool.

Before proceeding, ensure that **OpenFOAM** is installed on your system. Installation instructions can be found on the [official OpenFOAM website](https://openfoam.org/download/).

## **Simulation Setup Guide**

### **1. Generate the Geometry**
- Define your parameters and generate the spheroid trap geometry via the **web interface** or by running the provided command-line tool.
- The generated **STL file** will represent the 3D shape of the trap.

### **2. Prepare for Simulation**
- Copy the generated **STL file** into the following directory:  
  ```sh
  /simulation/openFoam/constant/geometry
  ```
- Modify the **vertices** in the **blockMeshDict** file to ensure the mesh properly fits around your geometry:  
  ```sh
  /simulation/openFoam/system/blockMeshDict
  ```

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
