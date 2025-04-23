# Spheroid-Trap-Designer

<p align="center">
<img src="https://www.cda.cit.tum.de/research/microfluidics/logo-microfluidics-toolkit.png" style="margin:auto;width:60%"/>
</p>

The MMFT spheroid-trap designer is a tool that generates a microfluidic chip and/or channel design for trapping spheroids based on hydrodynamic principles in a dedicated location on a chip, by employing a SAT solver to optimize the design process and automatically compute the required parameters while adhering to ISO guidelines. It is possible to chain several traps together and thereby trap several spheroids. These traps are especially useful for studying complex biological processes such as the vascularization of spheroids and organoids [1].

The tool is available online: [https://www.cda.cit.tum.de/app/mmft-spheroid-trap-designer/](https://www.cda.cit.tum.de/app/mmft-TODO/).
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

## References

[[1]](https://doi.org/10.1038/s41467-024-45710-4) Quintard, C., Tubbs, E., Jonsson, G. et al. A microfluidic platform integrating functional vascularized organoids-on-chip. Nat Commun 15, 1452 (2024).

**Developed in collaboration between:**


<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/cda-tum/mmft-spheroid-trap-designer/main/interface/tum_x_ceaLeti.png" width="90%">
  </picture>
</p>