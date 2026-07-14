# DCE-MRI Pharmacokinetic Analysis — Population AIF for Rat Brain

Python implementation of the extended Tofts model for DCE-MRI pharmacokinetic
analysis, including derivation and validation of a population-based vascular/
arterial input function (AIF) for rat brain at 9.4T.

Used in the paper:
**"Quantitative Pharmacokinetic Analysis by DCE-MRI in Preclinical Models:
Implementation of the Tofts Model and Derivation of a Population Vascular
Input Function for Rat Brain at 9.4T"**

## Requirements
- Python 3.x
- numpy, nibabel, matplotlib, scipy

## Usage
Place `DCE.nii`, `T1.nii`, and (optionally) `AIF.nii` in the working
directory and run the script. See inline comments for parameter details.

## Output
- `Ktrans_map.nii`, `ve_map.nii`, `kep_map.nii`, `r2_map.nii`
- `AIF.png`, `pharmacokinetic_maps.png`
- `results.txt` with summary statistics
