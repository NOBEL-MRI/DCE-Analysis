

import os
import numpy as np
import nibabel as nib
import matplotlib
import sys
if "matplotlib.backends" not in sys.modules:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from scipy.optimize import curve_fit
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr

# ==========================================
# PARAMETERS
# ==========================================
FA          = np.deg2rad(15.0)
TR          = 0.022
r1          = 4.2
dt          = 2.816
HEMATOCRIT  = 0.45

BASELINE_START = None
BASELINE_END   = None
ARRIVAL_FRAME  = None

MAX_CONTRA_PIXELS = 1000
KTRANS_MIN  = 1e-7
KTRANS_MAX  = 2.0
VE_MIN      = 0.02
VE_MAX      = 1.0
T1_CLIP_MIN = 0.3
T1_CLIP_MAX = 5.0

P0_LIST = [
    [0.0002, 0.2],
    [0.001,  0.05],
    [0.01,   0.3],
]

# ==========================================
# 1. CARGA DATOS
# ==========================================
print("=" * 50)
print("DCE-MRI Pharmacokinetic Analysis")
print("=" * 50)
print("\nLoading NII Data...")

dce_img = nib.load("DCE.nii")
t1_img  = nib.load("T1.nii")
dce_raw = dce_img.get_fdata()
t1_raw  = t1_img.get_fdata()

USE_MEASURED_AIF = os.path.exists("AIF.nii")
if USE_MEASURED_AIF:
    aif_img = nib.load("AIF.nii")
    aif_raw = aif_img.get_fdata()
    print("  AIF.nii found — Measured AIF will be used")
else:
    aif_img = None
    aif_raw = None
    print("  AIF.nii not found")

# ==========================================
# ORIENTATION
# ==========================================
print("\n--- Checking orientation ---")

def orientation_code(img):
    return "".join(nib.aff2axcodes(img.affine))

for name, img in [("DCE", dce_img), ("T1", t1_img)] + ([("AIF", aif_img)] if USE_MEASURED_AIF else []):
    code = orientation_code(img)
    det  = np.linalg.det(img.affine[:3, :3])
    flip_warn = " WARNING negative determinant" if det < 0 else ""
    print(f"  {name}: orientation={code}  det={det:.1f}{flip_warn}")

def affines_match(a, b, tol=1e-3):
    return np.allclose(a[:3, :3], b[:3, :3], atol=tol) and \
           np.allclose(a[:3,  3], b[:3,  3], atol=tol)

if not affines_match(dce_img.affine, t1_img.affine):
    print("  WARNING: T1 affine differs from DCE")
if USE_MEASURED_AIF and not affines_match(dce_img.affine, aif_img.affine):
    print("  WARNING: AIF affine differs from DCE")

dce_can = nib.as_closest_canonical(dce_img)
t1_can  = nib.as_closest_canonical(t1_img)
dce_raw = dce_can.get_fdata()
t1_raw  = t1_can.get_fdata()

if USE_MEASURED_AIF:
    aif_can = nib.as_closest_canonical(aif_img)
    aif_raw = aif_can.get_fdata()
    print(f"  Final orientation: {orientation_code(dce_can)} "
          f"(T1: {orientation_code(t1_can)}, AIF: {orientation_code(aif_can)})")
else:
    print(f"  Final orientation: {orientation_code(dce_can)} "
          f"(T1: {orientation_code(t1_can)})")

dce_img = dce_can

# ==========================================
# AXES REORDERING
# ==========================================
def _match_axes_to_dce(vol3d, target_shape_3d):
    from itertools import permutations
    for perm in permutations(range(3)):
        candidate = tuple(vol3d.shape[p] for p in perm)
        if candidate == tuple(target_shape_3d):
            if perm != (0, 1, 2):
                print(f"    Reordering axes {perm} to match DCE {target_shape_3d}")
            return np.transpose(vol3d, perm)
    print(f"  WARNING: No exact permutation found: vol={vol3d.shape} vs DCE={target_shape_3d}")
    return vol3d

dce = np.transpose(dce_raw, (1, 0, 2, 3))

t1_raw_3d = t1_raw.squeeze() if 1 in t1_raw.shape else t1_raw
if t1_raw_3d.ndim != 3:
    raise ValueError(f"T1 is not 3D after squeeze: T1={t1_raw.shape}")

nx_dce, ny_dce, nz_dce = dce.shape[:3]
t1 = _match_axes_to_dce(t1_raw_3d, (nx_dce, ny_dce, nz_dce))

if USE_MEASURED_AIF:
    aif_raw_3d = aif_raw.squeeze() if 1 in aif_raw.shape else aif_raw
    if aif_raw_3d.ndim != 3:
        raise ValueError(f"AIF is not 3D after squeeze: AIF={aif_raw.shape}")
    aif_mask = _match_axes_to_dce(aif_raw_3d, (nx_dce, ny_dce, nz_dce))
else:
    aif_mask = None

t1 = np.flip(t1, axis=1).copy()
print("  T1 corrected: vertical flip applied (y-axis)")

aif_shape_str = str(aif_mask.shape) if USE_MEASURED_AIF else "N/A"
print(f"  Final shapes — DCE: {dce.shape}  T1: {t1.shape}  AIF: {aif_shape_str}")

t1_p75 = np.percentile(t1[t1 > 0], 75)
if t1_p75 > 50:
    print(f"  T1 in ms (P75={t1_p75:.1f}) — converting to s")
    t1 = t1 / 1000.0
else:
    print(f"  T1 in s (P75={t1_p75:.3f}) — no conversion")

t1 = np.clip(t1, T1_CLIP_MIN, T1_CLIP_MAX)

nx, ny, nz, nt = dce.shape
t_s   = np.arange(nt) * dt
t_min = t_s / 60.0

print(f"  DCE:      {dce.shape}  (x,y,z,t)")
print(f"  T1:       {t1.shape}   mean={np.mean(t1[t1>T1_CLIP_MIN]):.3f}s")
print(f"  Time:   {t_min[0]:.3f}-{t_min[-1]:.3f} min  ({nt} frames)")

# ==========================================
# BASELINE
# ==========================================
def detect_baseline(dce_4d, threshold=0.02, consec=5, baseline_min=5):
    mean_signal = np.mean(dce_4d.reshape(-1, dce_4d.shape[-1]), axis=0)
    S0   = np.mean(mean_signal[:baseline_min])
    norm = (mean_signal - S0) / (S0 + 1e-8)
    arrival = None
    count   = 0
    for i in range(baseline_min, len(norm)):
        if abs(norm[i]) > threshold:
            count += 1
            if count >= consec:
                arrival = i - consec + 1
                break
        else:
            count = 0
    if arrival is None:
        return None, None, None
    arrival = max(arrival, baseline_min)
    return 0, arrival, arrival

if BASELINE_START is None or BASELINE_END is None or ARRIVAL_FRAME is None:
    print("\n--- Automatic baseline detection ---")
    bs, be, af = detect_baseline(dce, threshold=0.02, consec=5, baseline_min=20)
    if bs is None:
        print("  Retrying with more permissive parameters...")
        bs, be, af = detect_baseline(dce, threshold=0.01, consec=3, baseline_min=20)
    if bs is None:
        raise RuntimeError("Could not detect baseline. Please define manually.")
    BASELINE_START = bs
    BASELINE_END   = be
    ARRIVAL_FRAME  = af
    print(f"  BASELINE_START : {BASELINE_START}")
    print(f"  BASELINE_END   : {BASELINE_END}  ({t_min[BASELINE_END]:.2f} min)")
    print(f"  ARRIVAL_FRAME  : {ARRIVAL_FRAME}  ({t_min[ARRIVAL_FRAME]:.2f} min)")
else:
    print(f"\n  Manual baseline: frames {BASELINE_START}-{BASELINE_END}, arrival={ARRIVAL_FRAME}")

# ==========================================
# 2. SIGNAL -> CONCENTRATION CONVERSION
# ==========================================
def signal_to_ct(signal, t10_arr, is_aif=False):
    t10   = np.clip(t10_arr, T1_CLIP_MIN, T1_CLIP_MAX)
    E1    = np.exp(-TR / t10)
    Sstar = (1 - E1) / (1 - np.cos(FA) * E1)
    Sss   = np.mean(signal[:, BASELINE_START:BASELINE_END], axis=1)
    Sstar = Sstar[:, np.newaxis]
    Sss   = Sss[:,   np.newaxis]
    A  = 1 - np.cos(FA) * Sstar * signal / Sss
    B  = 1 - Sstar * signal / Sss
    AB = A / np.where(np.abs(B) < 1e-10, 1e-10, B)
    AB = np.where(AB > 0, AB, np.nan)
    AB = np.clip(AB, 1e-06, None)
    R1t   = (1.0 / TR) * np.log(AB)
    R1_ss = np.nanmean(R1t[:, BASELINE_START:BASELINE_END], axis=1, keepdims=True)
    R10   = (1.0 / t10)[:, np.newaxis]
    R1t   = R1t + (R10 - R1_ss)
    R1t   = np.where(np.isnan(R1t), R10, R1t)
    if is_aif:
        return (R1t - R10) / (r1 * (1 - HEMATOCRIT))
    else:
        return (R1t - R10) / r1

# ==========================================
# 3. AIF
# ==========================================
DOSE     = 0.3
AIF_A1   = 3.99
AIF_m1   = 10.17
AIF_A2   = 4.78
AIF_m2   = 0.171
T1_BLOOD = 1.9

_transforms = {
    '1': ('original',        lambda m: m.copy()),
    '2': ('fliplr',          lambda m: np.fliplr(m)),
    '3': ('flipud',          lambda m: np.flipud(m)),
    '4': ('flipud + fliplr', lambda m: np.flipud(np.fliplr(m))),
}

if USE_MEASURED_AIF:
    print("\n--- AIF measured (from AIF.nii) ---")
    xs_aif, ys_aif, zs_aif = np.where(aif_mask > 0)
    print(f"  AIF Voxels: {len(xs_aif)}")

    # Extract AIF signals directly from dce_raw using aif_raw coordinates
    # without any flip or transformation — "no flip" is the correct orientation
    _aif_raw_orig = nib.load("AIF.nii").get_fdata().squeeze()
    _dce_raw_orig = nib.load("DCE.nii").get_fdata()
    xs_aif_c, ys_aif_c, zs_aif_c = np.where(_aif_raw_orig > 0)
    print(f"   AIF Voxels: {len(xs_aif_c)}")
    aif_signals = _dce_raw_orig[xs_aif_c, ys_aif_c, zs_aif_c, :]

    # "# Diagnostic: AIF overlay on DCE in original (unprocessed) space
    mean_per_frame = np.mean(_dce_raw_orig.reshape(-1, _dce_raw_orig.shape[-1]), axis=0)
    _peak_frame    = int(np.argmax(mean_per_frame))
    _nz_orig       = _dce_raw_orig.shape[2]
    fig_diag2, axes_diag2 = plt.subplots(1, _nz_orig, figsize=(4 * _nz_orig, 4))
    if _nz_orig == 1:
        axes_diag2 = [axes_diag2]
    for z in range(_nz_orig):
        axes_diag2[z].imshow(_dce_raw_orig[:, :, z, _peak_frame], cmap='gray', origin='lower')
        _aif_z = _aif_raw_orig[:, :, z] if _aif_raw_orig.ndim == 3 else _aif_raw_orig[:, :, z]
        axes_diag2[z].imshow(
            np.ma.masked_where(_aif_z == 0, _aif_z),
            cmap='autumn', origin='lower', alpha=0.7)
        axes_diag2[z].set_title(f"DCE peak + AIF  z={z}", fontsize=9)
        axes_diag2[z].axis('off')
    plt.suptitle("Diagnostic AIF — original unprocessed space\nVerify that the AIF (red) is over the vessel", fontsize=10)
    plt.tight_layout()
    plt.savefig("diagnostic_aif_original.png", dpi=150)
    plt.close(fig_diag2)
    print("  Original AIF diagnostic saved to diagnostic_aif_original.png")
    import subprocess as _sp2
    try:
        if sys.platform == "win32":
            _sp2.Popen(["start", "diagnostic_aif_original.png"], shell=True)
        else:
            _sp2.Popen(["open" if sys.platform == "darwin" else "xdg-open",
                        "diagnostic_aif_original.png"])
    except Exception:
        pass
    aif_t1s     = np.full(len(xs_aif_c), T1_BLOOD)
    print(f"  Fixed blood T1: {T1_BLOOD} s (literature value, 9.4T)")

    AIF_TOP_N  = 5
    aif_ratios = (np.max(aif_signals[:, ARRIVAL_FRAME:], axis=1) /
                  (np.mean(aif_signals[:, :BASELINE_END], axis=1) + 1e-8))
    top_idx    = np.argsort(aif_ratios)[::-1][:AIF_TOP_N]
    print(f"  Top {AIF_TOP_N} voxels by ratio: {[f'({xs_aif_c[i]},{ys_aif_c[i]},{zs_aif_c[i]}) ratio={aif_ratios[i]:.2f}' for i in top_idx]}")

    cp_pixels = signal_to_ct(aif_signals[top_idx], aif_t1s[top_idx], is_aif=True)
    cp        = np.mean(cp_pixels, axis=0)
    aif_label = f"AIF measured (peak={np.max(cp):.4f} mM)"
    aif_title = "Arterial Input Function (measured)"

else:
    # Ask for animal type
    ANIMAL_TYPE = input("\nIs it a rat or mouse? (rat/mouse):  ").strip().lower()
    while ANIMAL_TYPE not in ('rat', 'mouse'):
        ANIMAL_TYPE = input("  Type 'rat' or 'mouse': ").strip().lower()
    print(f"  Animal: {ANIMAL_TYPE}")

    if ANIMAL_TYPE == 'mouse':
        print("\n--- Population AIF (Loveless et al., mouse) ---")
        print(f"  Dose: {DOSE} mmol/kg")
        cp = np.zeros(nt)
        for i in range(nt):
            t = t_min[i] - t_min[ARRIVAL_FRAME]
            if t >= 0:
                cp[i] = DOSE * (AIF_A1 * np.exp(-AIF_m1 * t) +
                                AIF_A2 * np.exp(-AIF_m2 * t))
        aif_label = f"Population AIF Loveless (peak={np.max(cp):.4f} mM)"
        aif_title = f"Population AIF Loveless mouse ({DOSE} mmol/kg)"
    else:

        print("\n  No AIF.nii. Options:")
        print("  1 = Population AIF (derived from 12 rats in this paper)")
        print("  2 = Use AIF.nii — re-run with AIF.nii in the folder")
        _aif_opt = input("  Choose option (1/2): ").strip()
        while _aif_opt not in ('1', '2'):
            _aif_opt = input("  Type 1 or 2: ").strip()

        if _aif_opt == '2':
            raise RuntimeError(
                "Place AIF.nii in the folder and re-run.")
        else:
            print("\n--- Rat's Population AIF (this paper) ---")
            POP_A1 = 0.357  # [mM]
            POP_m1 = 0.383  # [min-1]
            POP_A2 = 0.364  # [mM]
            POP_m2 = 0.029  # [min-1]

            cp = np.zeros(nt)
            for i in range(nt):
                t = t_min[i] - t_min[ARRIVAL_FRAME]
                if t >= 0:
                    cp[i] = (POP_A1 * np.exp(-POP_m1 * t) +
                             POP_A2 * np.exp(-POP_m2 * t))
            aif_label = f"Rat's Population AIF (peak={np.max(cp):.4f} mM)"
            aif_title = "Rat's Population AIF (n=12)"

print(f"  AIF Cp peak:  {np.max(cp):.4f} mM")
print(f"  AIF at the end: {cp[-1]:.4f} mM")

np.save("AIF_cp.npy", cp)
np.savetxt("BASELINE_END.txt", [BASELINE_END, ARRIVAL_FRAME], fmt="%d")
print("  AIF_cp.npy and BASELINE_END.txt saved")

plt.figure(figsize=(10, 4))
plt.plot(t_min, cp, label=aif_label, color='steelblue', linewidth=2)
plt.axvline(t_min[ARRIVAL_FRAME], color='r', linestyle=':', label='arrival')
plt.axvspan(t_min[BASELINE_START], t_min[BASELINE_END],
            alpha=0.15, color='green', label='baseline')
plt.axhline(0, color='k', linestyle=':', linewidth=0.5)
plt.legend(fontsize=8); plt.xlabel("Time (min)"); plt.ylabel("Cp (mM)")
plt.title(aif_title)
plt.tight_layout()
plt.savefig("AIF.png", dpi=150)
plt.close()

# ==========================================
# 4. TISSUE ROI
# ==========================================
class Selector:
    def __init__(self, ax):
        self.verts = None
        self.poly  = PolygonSelector(ax, self.onselect,
                                     props=dict(color='r', linewidth=2))
    def onselect(self, verts):
        if len(verts) >= 2 and verts[0] == (0.0, 0.0):
            return
        self.verts = verts

def draw_roi(image_xy, title):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image_xy.T, cmap="gray", origin="lower")
    ax.set_title(title + "\n[Draw ROI and close the window]"
                 "\n[Close WITHOUT drawing to skip this slice]", fontsize=9)
    sel = Selector(ax)
    plt.show(block=True)
    if sel.verts is None:
        return None
    xg, yg = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    pts    = np.vstack((xg.flatten(), yg.flatten())).T
    return Path(sel.verts).contains_points(pts).reshape((nx, ny))

fig, axes = plt.subplots(1, nz, figsize=(15, 4))
if nz == 1:
    axes = [axes]
for z in range(nz):
    axes[z].imshow(np.mean(dce[:,:,z,:], axis=-1).T, cmap='gray', origin='lower')
    axes[z].set_title(f"DCE mean z={z}")
plt.suptitle("General view — reference for drawing ROIs", fontsize=12)
plt.tight_layout()
plt.savefig("general_view.png", dpi=150)
plt.close("all")

import time
time.sleep(0.5)
try:
    from PyQt5.QtWidgets import QApplication
    QApplication.processEvents()
except Exception:
    pass

if os.path.exists("brain_masks.npy"):
    brain_masks = np.load("brain_masks.npy")
    print(f"  brain_masks.npy found — using existing ROI ({np.sum(brain_masks)} pixels)")
    for z in range(nz):
        print(f"  z={z}: {np.sum(brain_masks[:,:,z])} pixels")
else:
    print("Draw ROI manually...")
    print("  [Draw ROI on each slice. Close without drawing to skip that slice]")
    brain_masks = np.zeros((nx, ny, nz), dtype=bool)
    for z in range(nz):
        mean_img = np.mean(dce[:, :, z, :], axis=-1)
        mask_raw = draw_roi(mean_img, f"Tissue ROI — Slice z={z}")
        if mask_raw is None:
            print(f"  z={z}: no ROI — slice excluded")
            continue
        mask_raw  = binary_erosion(mask_raw, iterations=2)
        S0_map    = np.mean(dce[:, :, z, BASELINE_START:BASELINE_END], axis=-1) + 1e-8
        Speak_map = np.max(dce[:, :, z, ARRIVAL_FRAME:], axis=-1)
        brain_masks[:, :, z] = mask_raw & (Speak_map / S0_map > 1.02)
        print(f"  z={z}: {np.sum(brain_masks[:,:,z])} pixels included")
    np.save("brain_masks.npy", brain_masks)
    print("ROI saved in brain_masks.npy")

# ==========================================
# ARTIFACT EXCLUSION
# ==========================================
CV_K_MAD = 5.0

n_antes_art = np.sum(brain_masks)
for z in range(nz):
    if np.sum(brain_masks[:, :, z]) == 0:
        continue
    senal_baseline = dce[:, :, z, BASELINE_START:BASELINE_END]
    media_b  = np.mean(senal_baseline, axis=-1) + 1e-8
    std_b    = np.std(senal_baseline, axis=-1)
    cv_map   = std_b / media_b
    cv_roi    = cv_map[brain_masks[:, :, z]]
    mediana   = np.median(cv_roi)
    mad       = np.median(np.abs(cv_roi - mediana))
    umbral_cv = mediana + CV_K_MAD * mad
    brain_masks[:, :, z] = brain_masks[:, :, z] & (cv_map <= umbral_cv)

n_excl_art = n_antes_art - np.sum(brain_masks)
print(f"  Artifact exclusion (adaptive CV, k={CV_K_MAD}): {n_excl_art} pixels ({100*n_excl_art/max(n_antes_art,1):.1f}%)")

# ==========================================
# CSF EXCLUSION
# ==========================================
T1_LCR_THRESHOLD = 3.5

n_antes  = np.sum(brain_masks)
lcr_mask = t1 > T1_LCR_THRESHOLD
brain_masks = brain_masks & ~lcr_mask
n_despues   = np.sum(brain_masks)
n_excluidos = n_antes - n_despues

print(f"\n  Ventricle/CSF exclusion (T1>{T1_LCR_THRESHOLD}s):")
print(f"  Pixels before : {n_antes}")
print(f"  CSF pixels    : {n_excluidos} ({100*n_excluidos/n_antes:.1f}%)")
print(f"  Pixels after  : {n_despues}")

print(f"\nTotal pixels: {np.sum(brain_masks)}")
for z in range(nz):
    print(f"  z={z}: {np.sum(brain_masks[:,:,z])}")

# ==========================================
# CONTRALATERAL ROI
# ==========================================
use_contralateral = input("\nCalculate contralateral ROI? (y/n): ").strip().lower()
contralateral_masks = None

if use_contralateral == 'y':
    MAX_ROI_PIXELS = MAX_CONTRA_PIXELS
    total_roi = np.sum(brain_masks)
    if total_roi > MAX_ROI_PIXELS:
        print(f"  ROI has {total_roi} pixels (>{MAX_ROI_PIXELS}) — applying erosion...")
        brain_masks_contra = brain_masks.copy()
        iter_erosion = 0
        while np.sum(brain_masks_contra) > MAX_ROI_PIXELS and iter_erosion < 20:
            new_mask = np.zeros_like(brain_masks_contra)
            for z in range(nz):
                if np.sum(brain_masks_contra[:, :, z]) > 0:
                    new_mask[:, :, z] = binary_erosion(brain_masks_contra[:, :, z], iterations=1)
            brain_masks_contra = new_mask
            iter_erosion += 1
        print(f"  Trimmed ROI: {np.sum(brain_masks_contra)} pixels after {iter_erosion} erosions")
    else:
        brain_masks_contra = brain_masks.copy()

    xs_roi, ys_roi = np.where(np.any(brain_masks_contra, axis=2))
    cx_orig = int(np.mean(xs_roi))
    cy_orig = int(np.mean(ys_roi))
    print(f"  ROI Centroid: ({cx_orig}, {cy_orig})")

    _z_contra = [z for z in range(nz) if np.sum(brain_masks[:,:,z]) > 0]
    _z_show   = _z_contra[len(_z_contra)//2]

    fig_click, ax_click = plt.subplots(figsize=(7, 7))
    ax_click.imshow(_display(np.mean(dce[:, :, _z_show, :], axis=-1)), cmap='gray')
    roi_orig = np.ma.masked_where(~brain_masks[:, :, _z_show], np.ones((nx, ny)))
    ax_click.imshow(_display(roi_orig), cmap='Reds', alpha=0.5)
    ax_click.set_title("Click on the center of the contralateral hemisphere", fontsize=9)
    ax_click.axis('off')

    clicked = []
    def onclick(event):
        if event.inaxes and len(clicked) == 0:
            clicked.append((event.xdata, event.ydata))
            ax_click.plot(event.xdata, event.ydata, 'b+', markersize=20, markeredgewidth=3)
            fig_click.canvas.draw()

    fig_click.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show(block=True)

    if len(clicked) == 0:
        print("  No click detected — contralateral cancelled")
        contralateral_masks = None
    else:
        col_click, row_click = clicked[0]
        if _choice == '1':
            cx_data = int(row_click); cy_data = int(col_click)
        elif _choice == '2':
            cx_data = int(row_click); cy_data = int(nx - 1 - col_click)
        elif _choice == '3':
            cx_data = int(ny - 1 - row_click); cy_data = int(col_click)
        elif _choice == '4':
            cx_data = int(ny - 1 - row_click); cy_data = int(nx - 1 - col_click)

        dx = cx_data - cx_orig
        dy = cy_data - cy_orig
        print(f"  Displacement: ({dx}, {dy})")

        contralateral_masks = np.zeros_like(brain_masks)
        for z in range(nz):
            for x, y in zip(*np.where(brain_masks_contra[:, :, z])):
                nx_new = x + dx
                ny_new = y + dy
                if 0 <= nx_new < nx and 0 <= ny_new < ny:
                    contralateral_masks[nx_new, ny_new, z] = True

    nz_drawn = [z for z in range(nz) if np.sum(brain_masks[:,:,z]) > 0]
    ncols    = max(len(nz_drawn), 1)
    fig_contra, axes_contra = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
    if ncols == 1:
        axes_contra = axes_contra[:, np.newaxis]

    for idx, z in enumerate(nz_drawn):
        dce_mean = np.mean(dce[:, :, z, :], axis=-1)
        axes_contra[0, idx].imshow(_display(dce_mean), cmap='gray')
        roi_orig_v = np.ma.masked_where(~brain_masks[:, :, z], np.ones((nx, ny)))
        axes_contra[0, idx].imshow(_display(roi_orig_v), cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        axes_contra[0, idx].set_title(f"Original ROI z={z}", fontsize=9)
        axes_contra[0, idx].axis('off')
        axes_contra[1, idx].imshow(_display(dce_mean), cmap='gray')
        roi_contra_v = np.ma.masked_where(~contralateral_masks[:, :, z], np.ones((nx, ny)))
        axes_contra[1, idx].imshow(_display(roi_contra_v), cmap='Blues', alpha=0.4, vmin=0, vmax=1)
        axes_contra[1, idx].set_title(f"Contralateral ROI z={z}", fontsize=9)
        axes_contra[1, idx].axis('off')

    plt.suptitle("Red = Original ROI  |  Blue = Contralateral ROI", fontsize=10)
    plt.tight_layout()
    plt.savefig("contralateral_check.png", dpi=150)
    plt.close(fig_contra)
    print("  Image saved in contralateral_check.png")

    try:
        import subprocess as _sp
        if sys.platform == "win32":
            _sp.Popen(["start", "contralateral_check.png"], shell=True)
        elif sys.platform == "darwin":
            _sp.Popen(["open", "contralateral_check.png"])
        else:
            _sp.Popen(["xdg-open", "contralateral_check.png"])
    except Exception:
        pass

    confirm = input("\nDoes the contralateral look correct? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  Contralateral discarded")
        contralateral_masks = None
    else:
        print(f"  Contralateral confirmed — {np.sum(contralateral_masks)} pixels")

# ==========================================
# 5. TOFTS MODEL
# ==========================================
def build_conv_fn(t, cp):
    T_diff = np.maximum(t[:, np.newaxis] - t[np.newaxis, :], 0)
    tri    = np.tril(np.ones((len(t), len(t)), dtype=bool))
    def conv(kep):
        return np.trapz(
            np.where(tri, np.exp(-kep * T_diff) * cp[np.newaxis, :], 0.0),
            t, axis=1)
    return conv

conv_fn = build_conv_fn(t_min, cp)

def tofts_pred(t, Ktrans, ve):
    return Ktrans * conv_fn(Ktrans / (ve + 1e-6))

_SMOOTH_W = 5
_HALF_W   = _SMOOTH_W // 2

def smooth(ct):
    kernel = np.ones(_SMOOTH_W) / _SMOOTH_W
    raw    = np.convolve(ct, kernel, mode='same')
    for i in range(_HALF_W):
        raw[i]        = np.mean(ct[:i + _HALF_W + 1])
        raw[-(i + 1)] = np.mean(ct[-(i + _HALF_W + 1):])
    return raw

def fit_pixel(ct_raw):
    ct = np.asarray(ct_raw, dtype=float)
    if not np.all(np.isfinite(ct)):
        return None
    if np.max(ct) <= 0 or np.std(ct) < 1e-6:
        return None
    ct = smooth(ct)
    if not np.all(np.isfinite(ct)):
        return None
    best_result = None
    best_sse    = np.inf
    for p0 in P0_LIST:
        try:
            p, _ = curve_fit(
                tofts_pred, t_min, ct,
                p0=p0,
                bounds=([KTRANS_MIN, VE_MIN], [KTRANS_MAX, VE_MAX]),
                maxfev=5000,
                ftol=1e-8,
                xtol=1e-6
            )
            pred = tofts_pred(t_min, p[0], p[1])
            if not np.all(np.isfinite(pred)):
                continue
            sse = np.sum((ct - pred) ** 2)
            if sse < best_sse:
                best_sse    = sse
                best_result = (p[0], p[1], p[0] / (p[1] + 1e-6))
        except (RuntimeError, ValueError):
            continue
    return best_result

# ==========================================
# 6. FITTING
# ==========================================
K_map   = np.zeros((nx, ny, nz))
ve_map  = np.zeros_like(K_map)
kep_map = np.zeros_like(K_map)
r2_map  = np.zeros_like(K_map)

fitting_masks = brain_masks.copy()
if contralateral_masks is not None:
    fitting_masks = fitting_masks | contralateral_masks

print("\n--- Pharmacokinetic fitting ---")
for z in range(nz):
    xs, ys = np.where(fitting_masks[:, :, z])
    if len(xs) == 0:
        continue
    print(f"\nSlice z={z}: {len(xs)} pixels...")
    signals  = dce[xs, ys, z, :]
    t10s     = t1[xs, ys, z]
    ct_batch = signal_to_ct(signals, t10s, is_aif=False)
    n_ok = 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        result = fit_pixel(ct_batch[i])
        if result is None:
            continue
        Ktrans, ve, kep = result
        ct_sm  = smooth(ct_batch[i])
        pred   = tofts_pred(t_min, Ktrans, ve)
        ss_res = np.sum((ct_sm - pred) ** 2)
        ss_tot = np.sum((ct_sm - np.mean(ct_sm)) ** 2)
        K_map[x, y, z]   = Ktrans
        ve_map[x, y, z]  = ve
        kep_map[x, y, z] = kep
        r2_map[x, y, z]  = 1 - ss_res / (ss_tot + 1e-10)
        n_ok += 1
    print(f"  Converged: {n_ok}/{len(xs)} ({100*n_ok/len(xs):.1f}%)")

# ==========================================
# 7. NIfTI SAVING
# ==========================================
ref_affine = dce_img.affine
ref_header = dce_img.header.copy()

def save_nii(data_xyz, filename):
    out = nib.Nifti1Image(np.transpose(data_xyz, (1, 0, 2)),
                          affine=ref_affine, header=ref_header)
    nib.save(out, filename)
    print(f"  Saved: {filename}")

print("\n--- Saving NIfTI maps ---")
save_nii(K_map,                           "Ktrans_map.nii")
save_nii(ve_map,                          "ve_map.nii")
save_nii(np.clip(kep_map, 0, KTRANS_MAX), "kep_map.nii")
save_nii(r2_map,                          "r2_map.nii")

# ==========================================
# 8. STATISTICS
# ==========================================
R2_MIN  = 0.2
mask_ok = brain_masks & (K_map > 1e-6) & (r2_map >= R2_MIN)

print(f"\n  R2>={R2_MIN} filter: {np.sum(mask_ok)} valid pixels " f"({100*np.sum(mask_ok)/np.sum(brain_masks & (K_map>1e-6)):.1f}% of total converged)")

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)

def stats(name, m, unit=""):
    v = m[mask_ok]
    if len(v) == 0:
        print(f"\n{name}: no converged pixels"); return
    print(f"\n{name} {unit}")
    print(f"  N pixels : {len(v)}")
    print(f"  Mean     : {np.mean(v):.5f}")
    print(f"  Median   : {np.median(v):.5f}")
    print(f"  Std       : {np.std(v):.5f}")
    print(f"  P5-P95    : {np.percentile(v,5):.5f} - {np.percentile(v,95):.5f}")

stats("Ktrans", K_map,   "[min^-1]")
stats("ve",     ve_map)
stats("kep",    kep_map, "[min^-1]")
print(f"\nR2 mean: {np.mean(r2_map[mask_ok]):.3f}  median: {np.median(r2_map[mask_ok]):.3f}")

ve_low  = mask_ok & (ve_map < 0.1)
ve_high = mask_ok & (ve_map > 0.9)
print("\nSubgroups ve:")
if np.sum(ve_low) > 0:
    print(f"  ve<0.1: n={np.sum(ve_low):5d}  Kt median={np.median(K_map[ve_low]):.5f} min^-1")
if np.sum(ve_high) > 0:
    print(f"  ve>0.9: n={np.sum(ve_high):5d}  Kt median={np.median(K_map[ve_high]):.5f} min^-1")

print("\nBounds hits:")
bm = fitting_masks
print(f"  Kt<1e-6:   {np.sum(K_map[bm]<1e-6):5d} ({100*np.mean(K_map[bm]<1e-6):.1f}%)")
print(f"  Kt>=2.0:   {np.sum(K_map[bm]>=1.999):5d} ({100*np.mean(K_map[bm]>=1.999):.1f}%)")
print(f"  ve<=VE_MIN:{np.sum(ve_map[bm]<=VE_MIN+0.001):5d} ({100*np.mean(ve_map[bm]<=VE_MIN+0.001):.1f}%)")
print(f"  ve>=1.0:   {np.sum(ve_map[bm]>=0.999):5d} ({100*np.mean(ve_map[bm]>=0.999):.1f}%)")

if contralateral_masks is not None:
    mask_ok_contra = contralateral_masks & (K_map > 1e-6) & (r2_map >= R2_MIN)
    print("\n" + "=" * 50)
    print("CONTRALATERAL RESULTS")
    print("=" * 50)

    def stats_contra(name, m, unit=""):
        v = m[mask_ok_contra]
        if len(v) == 0:
            print(f"\n{name}: no converged pixels"); return
        print(f"\n{name} {unit}")
        print(f"  N pixels : {len(v)}")
        print(f"  Mean     : {np.mean(v):.5f}")
        print(f"  Median   : {np.median(v):.5f}")
        print(f"  Std       : {np.std(v):.5f}")
        print(f"  P5-P95    : {np.percentile(v,5):.5f} - {np.percentile(v,95):.5f}")

    print(f"\n  Filter R2>={R2_MIN}: {np.sum(mask_ok_contra)} valid pixels")
    stats_contra("Ktrans", K_map,   "[min^-1]")
    stats_contra("ve",     ve_map)
    stats_contra("kep",    kep_map, "[min^-1]")
    print(f"\nR2 mean: {np.mean(r2_map[mask_ok_contra]):.3f}  median: {np.median(r2_map[mask_ok_contra]):.3f}")

    print("\n" + "=" * 50)
    print("ORIGINAL vs CONTRALATERAL COMPARISON")
    print("=" * 50)
    for name, m in [("Ktrans", K_map), ("ve", ve_map), ("kep", kep_map)]:
        v_orig   = m[mask_ok]
        v_contra = m[mask_ok_contra]
        if len(v_orig) > 0 and len(v_contra) > 0:
            print(f"\n{name}:")
            print(f"  Original      median={np.median(v_orig):.5f}  P5-P95={np.percentile(v_orig,5):.5f}-{np.percentile(v_orig,95):.5f}")
            print(f"  Contralateral median={np.median(v_contra):.5f}  P5-P95={np.percentile(v_contra,5):.5f}-{np.percentile(v_contra,95):.5f}")

if os.path.exists("DCE_tofts_fit_Ktrans.nii"):
    ktrans_ref = np.transpose(nib.load("DCE_tofts_fit_Ktrans.nii").get_fdata(), (1, 0, 2))
    v_py  = K_map[mask_ok];      v_py  = v_py[v_py > 1e-6]
    v_mat = ktrans_ref[mask_ok]; v_mat = v_mat[v_mat > 0]
    print("\nKtrans comparison Python vs MATLAB:")
    print(f"  Python — median: {np.median(v_py):.5f}  mean: {np.mean(v_py):.5f} min^-1")
    print(f"  MATLAB — median: {np.median(v_mat):.5f}  mean: {np.mean(v_mat):.5f} min^-1")

# ==========================================
# TXT SAVING
# ==========================================
try:
    with open("results.txt", "w", encoding="utf-8") as _f:
        _f.write("RESULTS DCE-MRI\n")
        _f.write("=" * 50 + "\n\n")
        _f.write(f"Filter R2>={R2_MIN}: {np.sum(mask_ok)} valid pixels\n\n")
        for _name, _m in [("Ktrans [min-1]", K_map), ("ve", ve_map), ("kep [min-1]", kep_map)]:
            _vv = _m[mask_ok]
            if len(_vv) == 0:
                continue
            _f.write(f"{_name}\n")
            _f.write(f"  N pixels : {len(_vv)}\n")
            _f.write(f"  Mean     : {np.mean(_vv):.5f}\n")
            _f.write(f"  Median   : {np.median(_vv):.5f}\n")
            _f.write(f"  Std       : {np.std(_vv):.5f}\n")
            _f.write(f"  P5-P95    : {np.percentile(_vv,5):.5f} - {np.percentile(_vv,95):.5f}\n\n")
        _f.write(f"R2 mean: {np.mean(r2_map[mask_ok]):.3f}  median: {np.median(r2_map[mask_ok]):.3f}\n")
        if contralateral_masks is not None:
            _mask_c = contralateral_masks & (K_map > 1e-6) & (r2_map >= R2_MIN)
            _f.write("\nORIGINAL vs CONTRALATERAL COMPARISON\n")
            _f.write("=" * 50 + "\n")
            for _name, _m in [("Ktrans", K_map), ("ve", ve_map), ("kep", kep_map)]:
                _vo = _m[mask_ok]
                _vc = _m[_mask_c]
                if len(_vo) > 0 and len(_vc) > 0:
                    _f.write(f"\n{_name}:\n")
                    _f.write(f"  Original      median={np.median(_vo):.5f}  P5-P95={np.percentile(_vo,5):.5f}-{np.percentile(_vo,95):.5f}\n")
                    _f.write(f"  Contralateral median={np.median(_vc):.5f}  P5-P95={np.percentile(_vc,5):.5f}-{np.percentile(_vc,95):.5f}\n")
    print("Results saved in results.txt")
except Exception as _e:
    print(f"Could not save results.txt: {_e}")

# ==========================================
# 9. VISUALIZATION
# ==========================================
K_vis   = np.where(mask_ok, K_map,   0.)
ve_vis  = np.where(mask_ok, ve_map,  0.)
kep_vis = np.where(mask_ok, np.clip(kep_map, 0, 1.0), 0.)

fig, axes = plt.subplots(3, nz, figsize=(15, 9))
if nz == 1:
    axes = axes[:, np.newaxis]
maps_vis = [K_vis, ve_vis, kep_vis]
labels   = ["Ktrans [min^-1]", "ve", "kep [min^-1]"]
vmaxs    = [0.3, 1.0, 1.0]
cmaps    = ["hot", "jet", "jet"]

_z_test   = nz // 2
_dce_test = np.mean(dce[:, :, _z_test, :], axis=-1)

fig_ori, axes_ori = plt.subplots(1, 4, figsize=(16, 4))
for key, (name, fn) in _transforms.items():
    idx = int(key) - 1
    axes_ori[idx].imshow(fn(_dce_test), cmap='gray')
    axes_ori[idx].set_title(f"{key}: {name}", fontsize=10)
    axes_ori[idx].axis('off')
plt.suptitle("Choose the correct orientation (1-4)", fontsize=11)
plt.tight_layout()
plt.savefig("orientation_selector.png", dpi=150)
plt.close(fig_ori)

import subprocess
try:
    if sys.platform == "win32":
        subprocess.Popen(["start", "orientation_selector.png"], shell=True)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open",
                          "orientation_selector.png"])
except Exception:
    pass

_choice = input("\nWhich orientation is correct? (1/2/3/4): ").strip()
while _choice not in _transforms:
    _choice = input("  Type 1, 2, 3 or 4: ").strip()

_display_fn = _transforms[_choice][1]
print(f"  Selected orientation: {_transforms[_choice][0]}")

def _display(slice_2d):
    return _display_fn(slice_2d)

for row, (m, label, vmax, cmap) in enumerate(zip(maps_vis, labels, vmaxs, cmaps)):
    for z in range(nz):
        im = axes[row, z].imshow(_display(m[:, :, z]), cmap=cmap, vmin=0, vmax=vmax)
        axes[row, z].set_title(f"{label} z={z}", fontsize=7)
        axes[row, z].axis("off")
    plt.colorbar(im, ax=axes[row, -1], fraction=0.046, pad=0.04)

plt.suptitle("Pharmacokinetic maps — Reversible Tofts model", fontsize=11)
plt.tight_layout()
plt.savefig("pharmacokinetic_maps.png", dpi=150)
plt.show()
print("\nMaps saved in pharmacokinetic_maps.png")

if os.path.exists("DCE_tofts_fit_Ktrans.nii"):
    ktrans_ref = np.transpose(nib.load("DCE_tofts_fit_Ktrans.nii").get_fdata(), (1, 0, 2))
    both  = mask_ok & (ktrans_ref > 0)
    v_py  = K_map[both]
    v_mat = ktrans_ref[both]
    lim   = np.percentile(np.concatenate([v_mat, v_py]), 99)

    fig, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    axes2[0].scatter(v_mat, v_py, alpha=0.05, s=1, color='steelblue')
    axes2[0].plot([0, lim], [0, lim], 'r--', label="identity")
    axes2[0].set_xlabel("Ktrans MATLAB [min^-1]")
    axes2[0].set_ylabel("Ktrans Python [min^-1]")
    axes2[0].set_title("Pixel-by-pixel comparison")
    axes2[0].set_xlim(0, lim); axes2[0].set_ylim(0, lim)
    axes2[0].legend()

    v_py_all  = K_map[mask_ok]; v_py_all  = v_py_all[v_py_all > 1e-6]
    v_mat_all = ktrans_ref[mask_ok]; v_mat_all = v_mat_all[v_mat_all > 0]
    bins = np.linspace(0, np.percentile(np.concatenate([v_mat_all, v_py_all]), 98), 50)
    axes2[1].hist(v_mat_all, bins=bins, alpha=0.5, label="MATLAB", color='orange')
    axes2[1].hist(v_py_all,  bins=bins, alpha=0.5, label="Python",  color='steelblue')
    axes2[1].set_xlabel("Ktrans [min^-1]"); axes2[1].set_ylabel("N pixels")
    axes2[1].set_title("Ktrans distribution"); axes2[1].legend()
    plt.tight_layout()
    plt.savefig("scatter_ktrans.png", dpi=150)
    plt.show()
    print("Scatter plot saved to scatter_ktrans.png")

    r, _ = pearsonr(v_mat, v_py)
    print("Correlation:", r)