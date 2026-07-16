# DCE-MRI Pharmacokinetic Analysis — Population AIF for Rat Brain

Python implementation of the extended Tofts model for DCE-MRI pharmacokinetic
analysis, including derivation and validation of a population-based vascular/
arterial input function (AIF) for rat brain at 9.4T.

Used in the paper:
**"Derivation of a Population-Based Vascular Input Function for the Rat Brain in DCE-MRI"**

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
