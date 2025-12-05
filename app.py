import streamlit as st
import pandas as pd
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---- STYLE sombre pour se fondre avec le thème Streamlit ----
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": "#FFFFFF",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#DDDDDD",
    "ytick.color": "#DDDDDD",
    "text.color": "#FFFFFF",
})

st.title("Comparaison : Modèle / TRACC")
st.markdown(
    """
    L’objectif de cette application est d’évaluer la précision de données météorologiques en les comparant à des données de référence, afin de juger de leur pertinence pour les projections climatiques futures en France.
    """,
    unsafe_allow_html=True
)

# -------- Paramètres --------
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
percentiles_list = [10, 25, 50, 75, 90]
couleur_modele = "goldenrod"
couleur_TRACC = "lightgray"
vmaxT = 5
vminT = -5
vmaxP = 100
vminP = 50
vmaxH = 100
vminH = -100
vmaxDJU = 150
vminDJU = -150

# -------- Noms des mois --------
mois_noms = {
    1: "01 - Janvier", 2: "02 - Février", 3: "03 - Mars",
    4: "04 - Avril", 5: "05 - Mai", 6: "06 - Juin",
    7: "07 - Juillet", 8: "08 - Août", 9: "09 - Septembre",
    10: "10 - Octobre", 11: "11 - Novembre", 12: "12 - Décembre"
}

# -------- Dossiers --------
dossiers = ["obs2000_2009", "obs", "typique"]

# Sélection du dossier
dossier_sel = st.selectbox("Choisir le dossier :", dossiers)

# Lister tous les fichiers .nc disponibles dans le dossier sélectionné
all_files = [f for f in os.listdir(dossier_sel) if f.endswith(".nc")]

# Menu pour choisir le fichier NetCDF
file_sel = st.selectbox("Choisir le fichier NetCDF (ville + code) :", all_files)

# Chemin du fichier sélectionné
nc_path = os.path.join(dossier_sel, file_sel)

# Déterminer les années disponibles
if "obs2000_2009" in dossier_sel:
    annees_dispo = list(range(2000, 2010))
elif "obs" in dossier_sel:
    annees_dispo = list(range(2010, 2020))
else:
    annees_dispo = [9999]  # Placeholder pour typique

# Choix de l'année
annee_sel = st.selectbox("Choisir l'année :", annees_dispo)

# Ouvrir le fichier
ds = xr.open_dataset(nc_path, decode_times=True)

# Extraire uniquement l'année sélectionnée (sauf typique)
if "typique" not in dossier_sel:
    mask = ds["time"].dt.year == annee_sel
    obs_time = ds["time"].values[mask]
    obs_temp = ds["T"].values[mask]
else:
    obs_time = ds["time"].values
    obs_temp = ds["T"].values

# -------- Upload CSV modèle --------
uploaded = st.file_uploader("Déposer le fichier CSV du modèle (colonne unique T°C) :", type=["csv"])

if uploaded:
    st.markdown("")

    # Lecture CSV modèle
    model_values = pd.read_csv(uploaded, header=0).iloc[:, 0].values

    # -------- RMSE --------
    def rmse(a, b):
        min_len = min(len(a), len(b))
        a_sorted = np.sort(a[:min_len])
        b_sorted = np.sort(b[:min_len])
        return np.sqrt(np.nanmean((a_sorted - b_sorted) ** 2))

    # -------- Précision basée sur les écarts de percentiles --------
    def precision_ecarts_percentiles(a, b):
        if len(a) == 0 or len(b) == 0:
            return np.nan
        percentiles = np.arange(1, 100)
        pa = np.percentile(a, percentiles)
        pb = np.percentile(b, percentiles)

        diff_moyenne = np.mean(np.abs(pa - pb))
        scale = np.std(pb)

        if scale == 0:
            return 100.0

        score = 100 * (1 - diff_moyenne / (2 * scale))
        score = max(0, min(100, score))

        return round(score, 2)

    # -------- Boucle sur les mois --------
    results_rmse = []
    obs_mois_all = []
    start_idx_model = 0

    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        mod_mois = model_values[start_idx_model:start_idx_model + nb_heures]
        obs_mois_vals = obs_temp[(start_idx_model // 24):(start_idx_model // 24 + nb_heures // 24)]
        obs_mois_all.append(obs_mois_vals)
        val_rmse = rmse(mod_mois, obs_mois_vals)
        pct_precision = precision_ecarts_percentiles(mod_mois, obs_mois_vals)
        results_rmse.append({
            "Mois": mois,
            "RMSE (°C)": round(val_rmse, 2),
            "Précision percentile (%)": pct_precision
        })
        start_idx_model += nb_heures

    # -------- DataFrame final --------
    df_rmse = pd.DataFrame(results_rmse)
    df_rmse_styled = (
        df_rmse.style
        .background_gradient(subset=["Précision percentile (%)"], cmap="RdYlGn", vmin=vminP, vmax=vmaxP, axis=None)
        .format({"Précision percentile (%)": "{:.2f}", "RMSE (°C)": "{:.2f}"})
    )

    st.subheader("Précision du modèle : RMSE et précision via écarts des percentiles")
    st.dataframe(df_rmse_styled, hide_index=True)
