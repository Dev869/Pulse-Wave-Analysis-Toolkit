import sys, os, json, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pwv_multiframe import load_dicom_series
import pwv_visual_analysis
import time
from pwv_visual_analysis import (
    create_masks,
    extract_ecg_trace,
    enhance_doppler_region,
    extract_doppler_trace
)

# -- Settings persistence ------------------------------------------------------
SETTINGS_PATH = os.path.expanduser('~/.pwv_settings.json')

def load_all_settings():
    if os.path.exists(SETTINGS_PATH):
        try:
            return json.load(open(SETTINGS_PATH, 'r'))
        except json.JSONDecodeError:
            pass
    return {}

def save_all_settings(profiles):
    with open(SETTINGS_PATH, 'w') as f:
        json.dump(profiles, f, indent=2)

# -- Peak & Base Detection ----------------------------------------------------
def find_peaks_and_bases(trace, height_frac=0.5):
    thr = height_frac * np.nanmax(trace)
    peaks = [i for i in range(1, len(trace)-1)
             if trace[i] > trace[i-1] and trace[i] > trace[i+1] and trace[i] >= thr]
    bases = []
    for p in peaks:
        j = p
        while j > 0 and trace[j] >= trace[j-1]:
            j -= 1
        bases.append(j)
    return peaks, bases

# -- Initialize default settings -----------------------------------------------
def init_settings():
    defaults = {
        'pix_per_mm':      28.556,
        'time_pixels':     839,
        'time_seconds':    0.7,
        'ec_peak_frac':    0.5,
        'dp_peak_frac':    0.5,
        'ec_diff_frac':    0.2,
        'dp_diff_frac':    0.2,
        'tt_min_ms':       20.0,
        'tt_max_ms':       30.0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_settings()

# -- Page config ---------------------------------------------------------------
st.set_page_config(page_title="PWV Analyzer", layout="wide")

# -- Top‐level tabs -----------------------------------------------------------
tab_settings, tab_analysis, tab_results = st.tabs(
    ["⚙️ Settings", "🖼️ Upload & Calibrate", "📥 Results / Download"]
)

# === SETTINGS TAB ============================================================
with tab_settings:
    st.title("Settings & Profiles")

    # Profile management
    profiles = load_all_settings()
    profile_names = list(profiles.keys())
    selected = st.selectbox("Load profile", ["<default>"] + profile_names)
    if st.button("Load Settings") and selected != '<default>':
        for k, v in profiles[selected].items():
            st.session_state[k] = v
        st.success(f"Loaded '{selected}' settings.")
        st.experimental_rerun()

    new_profile = st.text_input("Save current settings as:")
    if st.button("Save Settings"):
        if new_profile:
            profiles[new_profile] = {key: st.session_state[key] for key in [
                'pix_per_mm','time_pixels','time_seconds',
                'ec_peak_frac','dp_peak_frac','ec_diff_frac','dp_diff_frac',
                'tt_min_ms','tt_max_ms'
            ]}
            save_all_settings(profiles)
            st.success(f"Saved '{new_profile}' settings.")
        else:
            st.error("Enter a profile name before saving.")

    st.markdown("---")
    st.subheader("Calibration & Detection Thresholds")

    st.session_state.pix_per_mm = st.number_input(
        "Pixels per mm",
        min_value=0.1, max_value=1000.0,
        value=st.session_state.pix_per_mm
    )
    st.session_state.time_pixels = st.number_input(
        "Time scale bar pixels",
        min_value=1, max_value=10000,
        value=st.session_state.time_pixels
    )
    st.session_state.time_seconds = st.number_input(
        "Time scale bar seconds",
        min_value=0.001, max_value=10.0,
        value=st.session_state.time_seconds
    )
    sec_per_pix = st.session_state.time_seconds / st.session_state.time_pixels
    st.write(f"**Time scale:** {sec_per_pix*1000:.3f} ms/pixel")

    st.session_state.ec_peak_frac = st.number_input(
        "ECG peak fraction",
        min_value=0.0, max_value=1.0,
        value=st.session_state.ec_peak_frac, step=0.01
    )
    st.session_state.dp_peak_frac = st.number_input(
        "Doppler peak fraction",
        min_value=0.0, max_value=1.0,
        value=st.session_state.dp_peak_frac, step=0.01
    )
    st.session_state.ec_diff_frac = st.number_input(
        "Min ECG diff fraction",
        min_value=0.0, max_value=1.0,
        value=st.session_state.ec_diff_frac, step=0.01
    )
    st.session_state.dp_diff_frac = st.number_input(
        "Min Doppler diff fraction",
        min_value=0.0, max_value=1.0,
        value=st.session_state.dp_diff_frac, step=0.01
    )
    st.session_state.tt_min_ms = st.number_input(
        "Min TT to include (ms)",
        min_value=0.0, max_value=1000.0,
        value=st.session_state.tt_min_ms
    )
    st.session_state.tt_max_ms = st.number_input(
        "Max TT to include (ms)",
        min_value=1.0, max_value=1000.0,
        value=st.session_state.tt_max_ms
    )
    

# === ANALYSIS TAB ============================================================
with tab_analysis:
    st.title("Upload & Frame Selection")

    prox_file = st.file_uploader("Proximal DICOM", type=["dcm"])
    if prox_file:
        st.session_state["prox_basename"] = os.path.splitext(prox_file.name)[0]
    dist_file = st.file_uploader("Distal DICOM",   type=["dcm"])
    if dist_file:
        st.session_state["dist_basename"] = os.path.splitext(dist_file.name)[0]

    if prox_file and dist_file:
        tmpd = tempfile.mkdtemp()
        ppath = os.path.join(tmpd, 'prox.dcm')
        dpath = os.path.join(tmpd, 'dist.dcm')
        open(ppath, 'wb').write(prox_file.read())
        open(dpath, 'wb').write(dist_file.read())

        prox_stack = load_dicom_series(ppath)
        dist_stack = load_dicom_series(dpath)
        max_frames = min(len(prox_stack), len(dist_stack))
        frames_to_process = st.number_input(
            "Frames to process",
            min_value=1, max_value=max_frames,
            value=max_frames
        )

        st.markdown(
            "<style>.stButton>button {background-color:#28a745;color:white;}</style>",
            unsafe_allow_html=True
        )

# ─── Distance Picker ─────────────────────────────────────────────────────
    if st.button("Measure Separation"):
        # 1) Dump one frame from each stack
        tmp1 = os.path.join(tempfile.gettempdir(), "prox_frame.png")
        tmp2 = os.path.join(tempfile.gettempdir(), "dist_frame.png")

        # pick the same index (0 here) or whatever you want:
        frame_p = prox_stack[0]
        frame_d = dist_stack[0]

        # normalize to 0–255 uint8
        arr_p = ((frame_p - frame_p.min()) / (np.ptp(frame_p) + 1e-6) * 255).astype("uint8")
        arr_d = ((frame_d - frame_d.min()) / (np.ptp(frame_d) + 1e-6) * 255).astype("uint8")

        from PIL import Image
        Image.fromarray(arr_p).save(tmp1)
        Image.fromarray(arr_d).save(tmp2)

        # 2) Call the picker
        import subprocess, sys
        proc = subprocess.run(
            [sys.executable, "distance_tool.py", tmp1, tmp2],
            capture_output=True, text=True
        )

        # 3) Handle the result
        if proc.returncode == 0 and proc.stdout.startswith("Pixel distance:"):
            pix = float(proc.stdout.split(":")[1])
            mm  = pix / st.session_state.pix_per_mm
            st.session_state["probe_distance_mm"] = mm
            st.success(f"Measured separation: {mm:.2f} mm")
        else:
            st.error(f"Distance tool failed:\n{proc.stderr}")

        st.write("")  # spacing
# ────────────────────────────────────────────────────────────────────────────

    if st.button("Analyze"):
        start_time = time.time()
        total = frames_to_process
        progress_bar = st.progress(0)
        status_text = st.empty()
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        status_text.text(f"Analysis completed in {elapsed_time:.2f} seconds.")

        # run analysis and stash into session_state
        prox_records, dist_records = [], []
        frame_images_prox, frame_images_dist = [], []

        for i in range(frames_to_process):
            # update progress bar
            remaining = total - (i + 1)
            status_text.text(f"Processing frame {i+1} of {total} ({remaining} remaining)")
            progress_bar.progress((i + 1) / total)
            
            # --- proximal ---
            img_p = prox_stack[i]
            em_p, dr_p, _ = create_masks(img_p)
            ecg_p = extract_ecg_trace(em_p, em_p.shape[1])
            dm_p, ed_p, gr_p = enhance_doppler_region(dr_p)
            dop_p = extract_doppler_trace(dm_p, ed_p, gr_p, gr_p.shape[1])

            peaks_e, bases_e = find_peaks_and_bases(
                ecg_p, st.session_state.ec_peak_frac
            )
            valid_e = [
                (p,b) for p,b in zip(peaks_e,bases_e)
                if (ecg_p[p]-ecg_p[b]) >=
                    st.session_state.ec_diff_frac * np.nanmax(ecg_p)
            ]

            baseline_d = np.percentile(dop_p, 70)
            peaks_d, bases_d = find_peaks_and_bases(
                dop_p, st.session_state.dp_peak_frac
            )
            valid_d = [
                (p,b) for p,b in zip(peaks_d,bases_d)
                if (
                    dop_p[b] <= baseline_d and
                    (dop_p[p]-dop_p[b]) >=
                    st.session_state.dp_diff_frac * np.nanmax(dop_p)
                )
            ]

            for _, b in valid_e:
                next_bs = [bd for _, bd in valid_d if bd > b]
                if next_bs:
                    tt = abs((min(next_bs)-b) * sec_per_pix * 1000)
                    if (
                        st.session_state.tt_min_ms <= tt <=
                        st.session_state.tt_max_ms
                    ):
                        prox_records.append({'frame': i+1, 'TT_ms': tt})

            # save image
            fig, axs = plt.subplots(2,1,figsize=(6,4))
            axs[0].plot(ecg_p)
            axs[0].scatter([b for _,b in valid_e], [ecg_p[b] for _,b in valid_e], c='r')
            axs[0].set_title(f'Prox Frame {i+1}')
            axs[1].plot(dop_p)
            axs[1].scatter([b for _,b in valid_d], [dop_p[b] for _,b in valid_d], c='r')
            axs[1].axhline(baseline_d, color='k', linestyle='--')
            plt.tight_layout()
            fp = os.path.join(tmpd, f'prox_{i+1}.png')
            fig.savefig(fp); plt.close(fig)
            frame_images_prox.append(fp)

            # --- distal (same logic) ---
            img_d = dist_stack[i]
            em_d, dr_d, _ = create_masks(img_d)
            ecg_d = extract_ecg_trace(em_d, em_d.shape[1])
            dm_d, ed_d, gr_d = enhance_doppler_region(dr_d)
            dop_d = extract_doppler_trace(dm_d, ed_d, gr_d, gr_d.shape[1])

            peaks_e2, bases_e2 = find_peaks_and_bases(
                ecg_d, st.session_state.ec_peak_frac
            )
            valid_e2 = [
                (p,b) for p,b in zip(peaks_e2,bases_e2)
                if (ecg_d[p]-ecg_d[b]) >=
                    st.session_state.ec_diff_frac * np.nanmax(ecg_d)
            ]

            baseline_d2 = np.percentile(dop_d, 70)
            peaks_d2, bases_d2 = find_peaks_and_bases(
                dop_d, st.session_state.dp_peak_frac
            )
            valid_d2 = [
                (p,b) for p,b in zip(peaks_d2,bases_d2)
                if (
                    dop_d[b] <= baseline_d2 and
                    (dop_d[p]-dop_d[b]) >=
                    st.session_state.dp_diff_frac * np.nanmax(dop_d)
                )
            ]

            for _, b in valid_e2:
                next_bs2 = [bd for _,bd in valid_d2 if bd > b]
                if next_bs2:
                    tt2 = abs((min(next_bs2)-b) * sec_per_pix * 1000)
                    if (
                        st.session_state.tt_min_ms <= tt2 <=
                        st.session_state.tt_max_ms
                    ):
                        dist_records.append({'frame': i+1, 'TT_ms': tt2})

            # save image
            fig2, axs2 = plt.subplots(2,1,figsize=(6,4))
            axs2[0].plot(ecg_d)
            axs2[0].scatter([b for _,b in valid_e2], [ecg_d[b] for _,b in valid_e2], c='r')
            axs2[0].set_title(f'Dist Frame {i+1}')
            axs2[1].plot(dop_d)
            axs2[1].scatter([b for _,b in valid_d2], [dop_d[b] for _,b in valid_d2], c='r')
            axs2[1].axhline(baseline_d2, color='k', linestyle='--')
            plt.tight_layout()
            fd = os.path.join(tmpd, f'dist_{i+1}.png')
            fig2.savefig(fd); plt.close(fig2)
            frame_images_dist.append(fd)

        # Update progress bar
        status_text.text("All frames processed.")
        progress_bar.empty()
        
    
        # Build DataFrames
        df_prox = pd.DataFrame(prox_records)
        df_dist = pd.DataFrame(dist_records)
        dfp=pd.DataFrame(prox_records)
        dfd=pd.DataFrame(dist_records)

        # Build Results

        st.session_state['dfp_raw']     = dfp.copy()
        st.session_state['dfd_raw']     = dfd.copy()
        # give each widget a unique key and capture selections
        exclude_prox = st.multiselect(
            "Exclude Prox frames", 
            options=dfp['frame'].unique(), 
            key="exclude_prox"
        )
        exclude_dist = st.multiselect(
            "Exclude Dist frames", 
            options=dfd['frame'].unique(), 
            key="exclude_dist"
        )

        # Exclusion lists
        st.session_state['df_prox_filt'] = df_prox[~df_prox['frame'].isin(
            st.multiselect("Exclude Proximal frames", df_prox['frame'].unique())
        )]
        st.session_state['df_dist_filt'] = df_dist[~df_dist['frame'].isin(
            st.multiselect("Exclude Distal frames", df_dist['frame'].unique())
        )]

        # Summary
        avg_p = st.session_state['df_prox_filt']['TT_ms'].mean()
        avg_d = st.session_state['df_dist_filt']['TT_ms'].mean()
        dt = abs(avg_d - avg_p)
        dist_mm = st.session_state.get(
            "probe_distance_mm",
            1.0/ st.session_state.pix_per_mm
        )

        pwv = dist_mm / (dt/1000)/10 if dt > 0 else np.nan

        st.session_state['summary'] = {
            'avg_prox': avg_p,
            'avg_dist': avg_d,
            'dt': dt,
            'pwv': pwv
        }

        # Store images
        st.session_state['images_prox'] = frame_images_prox
        st.session_state['images_dist'] = frame_images_dist

        st.success("Analysis complete! Switch to the Results tab.")

# === RESULTS TAB =============================================================
with tab_results:
    st.title("Results & Downloads")
    if 'dfp_raw' in st.session_state and 'dfd_raw' in st.session_state:
        # Load DataFrames from session state
        base_p = st.session_state.get("prox_basename", "prox")
        base_d = st.session_state.get("dist_basename", "dist")
        df_p = st.session_state['dfp_raw']
        df_d = st.session_state['dfd_raw']
        summ = st.session_state['summary']

        st.subheader("Proximal Transit Times")
        st.dataframe(df_p)

        st.subheader("Distal Transit Times")
        st.dataframe(df_d)

        dmm = st.session_state.get("probe_distance_mm", 1.0 / st.session_state.pix_per_mm)

        # Calculate Average Transit Times
        avg_prox_tt = df_p['TT_ms'].mean()
        avg_dist_tt = df_d['TT_ms'].mean()

        # Calculate the difference between averages
        tt_difference = avg_prox_tt - avg_dist_tt

        st.subheader("Average Transit Times")
        st.write(f"**Measured Distance:** {dmm:.2f} mm")
        st.write(f"**Avg Prox TT ({base_p}):** {avg_prox_tt:.1f} ms")
        st.write(f"**Avg Dist TT ({base_d}):** {avg_dist_tt:.1f} ms")
        st.write(f"**ΔTT:** {tt_difference:.1f} ms")

        # Calculate PWV
        pwv = dmm / (abs(tt_difference) / 1000) / 10 if abs(tt_difference) > 0 else float('nan')
        st.write(f"**PWV:** {pwv:.1f} cm/s")

        # Prepare Full Results Table
        full_results = pd.DataFrame({
            'Frame Prox': df_p['frame'],
            'TT Prox (ms)': df_p['TT_ms'],
            'Frame Dist': df_d['frame'],
            'TT Dist (ms)': df_d['TT_ms']
        })

        # Append Summary Row at Top
        summary_data = pd.DataFrame({
            'Frame Prox': [''], 'TT Prox (ms)': [avg_prox_tt],
            'Frame Dist': [''], 'TT Dist (ms)': [avg_dist_tt],
            'TT Diff (ms)': [tt_difference],
            'Measured Distance (mm)': [dmm],
            'PWV (cm/s)': [pwv]
        })

        full_results = pd.concat([summary_data, full_results], ignore_index=True)

        st.subheader("Full Results Table")
        st.dataframe(full_results, use_container_width=True)

        # Download Full Results
        csv = full_results.to_csv(index=False)
        st.download_button(
            "Download Full Results CSV",
            csv,
            file_name=f"{base_p}_{base_d}_full_results.csv",
            mime="text/csv"
        )

        # Per-frame images
        st.subheader("Per-Frame Trace Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state['images_prox'], width=300)
        with col2:
            st.image(st.session_state['images_dist'], width=300)


    else:
        st.info("Run your analysis in the Upload & Calibrate tab first.")