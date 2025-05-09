import sys, os, json, tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numba
import pandas as pd
from scipy.signal import find_peaks
import streamlit as st
import matplotlib.pyplot as plt
from pwv_multiframe import load_dicom_series
import pwv_visual_analysis
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
import numba

@numba.njit
def compute_bases(peaks, trace):
    bases = np.empty(peaks.shape, np.int32)
    for idx in range(peaks.shape[0]):
        p = peaks[idx]
        j = p
        while j > 0 and trace[j] >= trace[j-1]:
            j -= 1
        bases[idx] = j
    return bases

def find_peaks_and_bases(trace, height_frac=0.5):
    thr = height_frac * np.nanmax(trace)
    peaks, _ = find_peaks(trace, height=thr)
    bases = compute_bases(peaks, trace)
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

# -- Tabs ----------------------------------------------------------------------
tab_settings, tab_analysis, tab_results = st.tabs(
    ["⚙️ Settings", "🖼️ Upload & Calibrate", "📥 Results / Download"]
)

# === SETTINGS TAB ============================================================
with tab_settings:
    st.title("Settings & Profiles")
    profiles = load_all_settings()
    sel = st.selectbox("Load profile", ["<default>"] + list(profiles.keys()))
    if st.button("Load Settings") and sel != '<default>':
        for k,v in profiles[sel].items(): st.session_state[k]=v
        st.success(f"Loaded '{sel}' settings.")
        st.experimental_rerun()
    newp = st.text_input("Save current settings as:")
    if st.button("Save Settings"):
        if newp:
            profiles[newp]={k:st.session_state[k] for k in [
                'pix_per_mm','time_pixels','time_seconds',
                'ec_peak_frac','dp_peak_frac','ec_diff_frac','dp_diff_frac',
                'tt_min_ms','tt_max_ms'
            ]}
            save_all_settings(profiles)
            st.success(f"Saved '{newp}' settings.")
        else:
            st.error("Enter a profile name before saving.")
    st.markdown("---")
    st.subheader("Calibration & Thresholds")
    st.session_state.pix_per_mm = st.number_input("Pixels per mm",0.1,1000.0,value=st.session_state.pix_per_mm)
    st.session_state.time_pixels = st.number_input("Time scale bar pixels",1,10000,value=st.session_state.time_pixels)
    st.session_state.time_seconds=st.number_input("Time scale bar seconds",0.001,10.0,value=st.session_state.time_seconds)
    sec_per_pix = st.session_state.time_seconds/st.session_state.time_pixels
    st.write(f"**Time scale:** {sec_per_pix*1000:.3f} ms/pixel")
    st.session_state.ec_peak_frac = st.number_input("ECG peak frac",0.0,1.0,value=st.session_state.ec_peak_frac,step=0.01)
    st.session_state.dp_peak_frac = st.number_input("Doppler peak frac",0.0,1.0,value=st.session_state.dp_peak_frac,step=0.01)
    st.session_state.ec_diff_frac = st.number_input("Min ECG diff frac",0.0,1.0,value=st.session_state.ec_diff_frac,step=0.01)
    st.session_state.dp_diff_frac = st.number_input("Min Doppler diff frac",0.0,1.0,value=st.session_state.dp_diff_frac,step=0.01)
    st.session_state.tt_min_ms     = st.number_input("Min TT include (ms)",0.0,1000.0,value=st.session_state.tt_min_ms)
    st.session_state.tt_max_ms     = st.number_input("Max TT include (ms)",1.0,1000.0,value=st.session_state.tt_max_ms)


# === ANALYSIS TAB ============================================================
with tab_analysis:
    st.title("Upload DICOM")
    prox_file = st.file_uploader("Proximal file (.dcm)",type=["dcm"])
    dist_file = st.file_uploader("Distal file (.dcm)",type=["dcm"])
    if prox_file and dist_file:
        # store basenames
        st.session_state.prox_basename = os.path.splitext(prox_file.name)[0]
        st.session_state.dist_basename = os.path.splitext(dist_file.name)[0]
        # write temp preserving ext
        tmpd = tempfile.mkdtemp()
        ext_p = os.path.splitext(prox_file.name)[1] or '.dcm'
        ext_d = os.path.splitext(dist_file.name)[1] or '.dcm'
        ppath = os.path.join(tmpd,f"prox{ext_p}")
        dpath = os.path.join(tmpd,f"dist{ext_d}")
        open(ppath,'wb').write(prox_file.read())
        open(dpath,'wb').write(dist_file.read())
        # load stacks
        prox_stack = load_dicom_series(ppath)
        dist_stack = load_dicom_series(dpath)
        # squeeze any singleton dims
        prox_stack = [np.squeeze(f) for f in prox_stack]
        dist_stack = [np.squeeze(f) for f in dist_stack]
        max_frames = min(len(prox_stack),len(dist_stack))
        frames_to_process = st.number_input("Frames to process",1,max_frames,value=max_frames)

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

        if st.button("Analyze"):
            total = frames_to_process
            status = st.empty()
            pbar   = st.progress(0.0)
            # capture settings
            ec_pf,dp_pf,ec_df,dp_df = (
                st.session_state.ec_peak_frac,
                st.session_state.dp_peak_frac,
                st.session_state.ec_diff_frac,
                st.session_state.dp_diff_frac
            )
            tt_min,tt_max,secpp = (
                st.session_state.tt_min_ms,
                st.session_state.tt_max_ms,
                st.session_state.time_seconds/st.session_state.time_pixels
            )
            prox_records,dist_records = [],[]
            def process_one(i):
                # proximal
                img = prox_stack[i]
                em,dr,_ = create_masks(img)
                ecg = extract_ecg_trace(em,em.shape[1])
                dm,ed,gr = enhance_doppler_region(dr)
                dop = extract_doppler_trace(dm,ed,gr,gr.shape[1])
                recp=[]
                pe,pb = find_peaks_and_bases(ecg,ec_pf)
                valid_e=[(p,b) for p,b in zip(pe,pb) if (ecg[p]-ecg[b])>=ec_df*np.nanmax(ecg)]
                bd = np.percentile(dop,70)
                pd,bd2 = find_peaks_and_bases(dop,dp_pf)
                valid_d=[(p,b) for p,b in zip(pd,bd2) if dop[b]<=bd and (dop[p]-dop[b])>=dp_df*np.nanmax(dop)]
                for _,b in valid_e:
                    nxt=[bd_ for _,bd_ in valid_d if bd_>b]
                    if nxt:
                        tt=abs((min(nxt)-b)*secpp*1000)
                        if tt_min<=tt<=tt_max:
                            recp.append({'frame':i+1,'TT_ms':tt})
                # distal
                img = dist_stack[i]
                em,dr,_ = create_masks(img)
                ecg = extract_ecg_trace(em,em.shape[1])
                dm,ed,gr = enhance_doppler_region(dr)
                dop = extract_doppler_trace(dm,ed,gr,gr.shape[1])
                recd=[]
                pe,pb = find_peaks_and_bases(ecg,ec_pf)
                valid_e=[(p,b) for p,b in zip(pe,pb) if (ecg[p]-ecg[b])>=ec_df*np.nanmax(ecg)]
                bd = np.percentile(dop,70)
                pd,bd2 = find_peaks_and_bases(dop,dp_pf)
                valid_d=[(p,b) for p,b in zip(pd,bd2) if dop[b]<=bd and (dop[p]-dop[b])>=dp_df*np.nanmax(dop)]
                for _,b in valid_e:
                    nxt=[bd_ for _,bd_ in valid_d if bd_>b]
                    if nxt:
                        tt=abs((min(nxt)-b)*secpp*1000)
                        if tt_min<=tt<=tt_max:
                            recd.append({'frame':i+1,'TT_ms':tt})
                return recp,recd
            # thread pool
            completed=0
            with ThreadPoolExecutor() as exe:
                futs=[exe.submit(process_one,i) for i in range(total)]
                for fut in as_completed(futs):
                    rp,rd = fut.result()
                    prox_records.extend(rp)
                    dist_records.extend(rd)
                    completed+=1
                    status.text(f"Frames remaining: {total-completed} of {total}")
                    pbar.progress(completed/total)
            status.text("Analysis complete!")
            pbar.empty()
            # DataFrames & exclusions
            dfp=pd.DataFrame(prox_records)
            dfd=pd.DataFrame(dist_records)
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
            st.session_state['df_prox_filt']=dfp[~dfp['frame'].isin(st.multiselect("Exclude Prox frames",dfp['frame'].unique()))]
            st.session_state['df_dist_filt']=dfd[~dfd['frame'].isin(st.multiselect("Exclude Dist frames",dfd['frame'].unique()))]
            # summary
            ap=st.session_state['df_prox_filt']['TT_ms'].mean()
            ad=st.session_state['df_dist_filt']['TT_ms'].mean()
            dt=abs(ad-ap)
            dcm=st.session_state.get("probe_distance_mm",1.0/st.session_state.pix_per_mm) / 10.0
            pwv=dcm/(dt/1000) if dt>0 else np.nan
            st.session_state['summary']={'avg_prox':ap,'avg_dist':ad,'dt':dt,'pwv':pwv}
            st.session_state['images_prox']=None
            st.session_state['images_dist']=None
            st.success("Done! Switch to Results tab.")

# === RESULTS TAB ============================================================
with tab_results:
    st.title("Results & Downloads")

    if 'dfp_raw' in st.session_state and 'dfd_raw' in st.session_state:
        base_p = st.session_state.get("prox_basename", "prox")
        base_d = st.session_state.get("dist_basename", "dist")
        dfp    = st.session_state['dfp_raw']
        dfd    = st.session_state['dfd_raw']
        summ   = st.session_state['summary']
        dmm    = st.session_state.get("probe_distance_mm", 1.0 / st.session_state.pix_per_mm)

        # 1) Merge per-cycle tables
        merged = pd.merge(
            dfp.rename(columns={'TT_ms': f'TT_ms_prox ({base_p})'}),
            dfd.rename(columns={'TT_ms': f'TT_ms_dist ({base_d})'}),
            on='frame', how='outer'
        ).sort_values('frame')

        st.subheader("Per-Cycle Transit Times")
        st.dataframe(merged, use_container_width=True)

        # 2) Display simple summary
        st.subheader("Summary")
        st.write(f"**Measured Distance:** {dmm:.2f} mm")
        st.write(f"**Avg Prox TT ({base_p}):** {summ['avg_prox']:.1f} ms")
        st.write(f"**Avg Dist TT ({base_d}):** {summ['avg_dist']:.1f} ms")
        st.write(f"**ΔTT:** {summ['dt']:.1f} ms")
        st.write(f"**PWV:** {summ['pwv']:.1f} cm/s")

        # 3) Build full results table with summary row at top
        # Define the extra summary columns
        summary_cols = [
            f"Avg TT Proximal ({base_p}) (ms)",
            f"Avg TT Distal ({base_d}) (ms)",
            "TT Diff (ms)",
            "Measured Distance (mm)",
            "PWV (cm/s)",
        ]
        # Full column order = merged.columns + summary_cols
        full_cols = list(merged.columns) + summary_cols

        # Build the single‐row summary DataFrame
        summary_data = {c: '' for c in merged.columns}
        summary_data.update({
            summary_cols[0]: summ['avg_prox'],
            summary_cols[1]: summ['avg_dist'],
            summary_cols[2]: summ['dt'],
            summary_cols[3]: dmm,
            summary_cols[4]: summ['pwv'],
        })
        summary_df = pd.DataFrame([summary_data], columns=full_cols)

        # Pad the per-cycle rows with empty summary columns
        cycles_df = merged.assign(**{c: '' for c in summary_cols})[full_cols]

        # Concatenate summary at top
        full = pd.concat([summary_df, cycles_df], ignore_index=True)

        st.subheader("Full Results Table")
        st.dataframe(full, use_container_width=True)

        # 4) Download
        csv = full.to_csv(index=False)
        st.download_button(
            "Download Full Results CSV",
            csv,
            file_name=f"{base_p}_{base_d}_full_results.csv",
            mime="text/csv"
        )

    else:
        st.info("Run analysis first in the Upload & Calibrate tab.")