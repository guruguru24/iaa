"""Pricing tool for the actuarial team (Streamlit).

    python -m src.tool --market w1                 # once: fit and save the final models (~20 min)
    streamlit run app/pricing_tool.py              # then open the tool

Single policy: expected 2025 cost (risk premium) with the GLM factor breakdown and the experience factor.
Batch: upload a CSV of policies (model feature columns; missing ones take typical values) and download the costs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import streamlit as st

from src.tool import glm_breakdown, load_bundle, price, to_frame

KEY_FACTORS = {
    "motor": ["driver_age", "ncb_years", "vehicle_value_group", "cover_type", "annual_mileage", "vehicle_use", "region", "voluntary_excess"],
    "buildings": ["sum_insured", "construction_type", "region", "flood_zone", "property_age", "occupancy", "voluntary_excess"],
    "contents": ["sum_insured", "tenure", "security_level", "high_value_items", "region", "voluntary_excess"],
}

st.set_page_config(page_title="Risk premium tool", layout="wide")


@st.cache_resource
def bundle():
    return load_bundle("w1")


try:
    B = bundle()
except FileNotFoundError:
    st.error("Model bundle not found. Run `python -m src.tool --market w1` first.")
    st.stop()

meta = B["meta"]
line = st.sidebar.selectbox("Line", list(B["lines"]))
LB = B["lines"][line]
glm = LB["models"]["glm"]
asm = meta["assumptions"]
st.sidebar.markdown(
    f"**Model** GLM + GBM `{LB['gbm_config']}` 50/50, experience modifier\n\n"
    f"**Trend** {LB['trend']:.1%} p.a. · **Cap** {LB['cap']:,} · **Weather** {asm['weather_mode']}\n\n"
    f"**Built** {meta['built']} · **git** `{meta['git']}`")


def widget(f, default, key):
    enc = glm.enc_
    if isinstance(default, bool):
        return st.checkbox(f, value=default, key=key)
    if f in enc.edges_ or default is None:
        return st.number_input(f, value=float(default) if default is not None else 0.0, key=key)
    options = [enc.base_[f]] + list(enc.levels_[f])
    options = [o for o in options if o != "missing"]
    if isinstance(default, float):                       # numeric with few values (e.g. excess): offer the levels
        vals = sorted({float(o) for o in options})
        return st.selectbox(f, vals, index=vals.index(default) if default in vals else 0, key=key)
    opts = sorted(set(options) | {default})
    return st.selectbox(f, opts, index=opts.index(default), key=key)


tab1, tab2, tab3 = st.tabs(["Single policy", "Batch (CSV)", "About"])

with tab1:
    st.subheader(f"{line.capitalize()} — single policy")
    c1, c2 = st.columns([1, 2])
    with c1:
        inputs = {"exposure": st.number_input("exposure (years)", 0.0027, 1.0, 1.0, step=0.25),
                  "hist_claims_1y": st.selectbox("claims in the last year", [0, 1, 2, 3], index=0)}
        for f in KEY_FACTORS[line]:
            if f in LB["defaults"]:
                inputs[f] = widget(f, LB["defaults"][f], f"k_{line}_{f}")
        with st.expander("All other factors (typical values pre-filled)"):
            for f in LB["factors"]:
                if f not in inputs:
                    inputs[f] = widget(f, LB["defaults"][f], f"o_{line}_{f}")
    res = price(LB, inputs).iloc[0]
    with c2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected cost", f"{res['expected_cost']:,.0f} CU")
        m2.metric("GLM", f"{res['glm']:,.0f}")
        m3.metric("GBM", f"{res['gbm']:,.0f}")
        m4.metric("Experience factor", f"{res['experience_factor']:.3f}")
        base, tab = glm_breakdown(glm, to_frame(LB, inputs))
        st.caption(f"GLM explanation: base {base:,.1f} CU per year × relativities below × exposure "
                   f"{inputs['exposure']:.2f} = {res['glm']:,.0f} CU. The final cost blends GLM and GBM 50/50 and "
                   f"applies the experience factor.")
        top = tab.head(12)
        st.bar_chart(top.set_index("factor")["relativity"])
        st.dataframe(tab.assign(value=tab["value"].astype(str), relativity=tab["relativity"].round(3)), hide_index=True, width="stretch")

with tab2:
    st.subheader(f"{line.capitalize()} — batch pricing")
    st.write("Upload a CSV with any of the model columns (missing columns take typical values; "
             "`exposure` defaults to 1, `hist_claims_1y` to 0).")
    st.code(", ".join(["exposure", "hist_claims_1y"] + LB["factors"]), language=None)
    up = st.file_uploader("Policies CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        out = pd.concat([df, price(LB, df).round(2)], axis=1)
        st.write(f"{len(out):,} policies · total expected cost {out['expected_cost'].sum():,.0f} CU")
        st.dataframe(out.head(200), width="stretch")
        st.download_button("Download priced CSV", out.to_csv(index=False).encode("utf-8"), f"priced_{line}.csv", "text/csv")

with tab3:
    st.subheader("Model, data and assumption versions")
    st.json({"built": meta["built"], "git": meta["git"], "market": meta["market"], "line": line,
             "gbm_config": LB["gbm_config"], "large_loss_cap": LB["cap"], "trend_2025": LB["trend"],
             "weather": asm["weather_mode"], "weather_years": asm.get("weather_years"),
             "experience_modifier": meta["experience_modifier"], "fit_info": LB["info"],
             "test_check": LB["test_check"]})
    st.markdown("Decisions behind each setting: `logs/decision_log.md` (G1 target, G2 data fixes, G3 trend & "
                "development, G4 model, G5 submissions). Risk premium only — no expenses, profit or reinsurance.")
