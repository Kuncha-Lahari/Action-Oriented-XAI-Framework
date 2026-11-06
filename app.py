"""
app.py
Interactive Streamlit front-end for the Action-Oriented XAI Framework.

Loads a case from the dataset (not from manual input), runs the full
pipeline for it, and displays prediction, counterfactual, evidence
retrieval, and action plan.

Default case selection prioritizes cases in [0.40, 0.50) so that the
dashboard opens on a case where the counterfactual search is highly
likely to succeed. The user can uncheck "Near-boundary only" to browse
the full denied set, including cases where the pipeline honestly fails.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import CONFIG
from loader import load_adult_income_data, DataLoaderResult
from predictor import train_and_select_model, PredictorResult
from causal_scm import discover_causal_structure
from counterfactual import generate_counterfactuals
from rag_retrieval import (
    build_resource_index,
    ONetClient,
    retrieve_resources_for_candidate,
)
from llm_synthesis import synthesize_action_plan


st.set_page_config(
    page_title="Action-Oriented XAI",
    page_icon="XAI",
    layout="wide",
)

st.title("Action-Oriented XAI Framework")
st.caption(
    "Dataset case → prediction → causal validation → counterfactual "
    "→ evidence retrieval → action plan"
)


@st.cache_resource(show_spinner=False)
def load_backend(force_synthetic: bool = False):
    data = load_adult_income_data(force_synthetic=force_synthetic)
    predictor = train_and_select_model(data)
    scm = discover_causal_structure(data)
    resource_index = build_resource_index()
    return data, predictor, scm, resource_index


@st.cache_resource(show_spinner=False)
def get_onet_client(api_key):
    return ONetClient(api_key=api_key or None)


def _predict_case(predictor, data, test_index):
    row = data.X_test.iloc[test_index]
    probability = float(
        predictor.best_model.predict_proba(
            pd.DataFrame([row])
        )[0, predictor.positive_class_label]
    )
    return row, probability


def _decode_change_row(feature, old, new, data):
    if feature == "education_num":
        old_label = data.education_label_for_years(float(old)) or ""
        new_label = data.education_label_for_years(float(new)) or ""
        return {
            "Feature": "years of education",
            "From": f"{int(round(old))} ({old_label})",
            "To": f"{int(round(new))} ({new_label})",
        }
    if feature in data.label_encoders:
        enc = data.label_encoders[feature]
        old_text = enc.inverse_transform([int(round(old))])[0]
        new_text = enc.inverse_transform([int(round(new))])[0]
        return {
            "Feature": feature.replace("_", " "),
            "From": str(old_text),
            "To": str(new_text),
        }
    return {
        "Feature": feature.replace("_", " "),
        "From": f"{old:.2f}",
        "To": f"{new:.2f}",
    }


def _priority(i, probs):
    """Prefer cases in bands most likely to yield a valid counterfactual."""
    p = probs[i]
    if 0.40 <= p < 0.50:
        return 0
    elif 0.30 <= p < 0.40:
        return 1
    elif 0.20 <= p < 0.30:
        return 2
    elif 0.10 <= p < 0.20:
        return 3
    else:
        return 4


with st.sidebar:
    st.header("Settings")
    force_synthetic = st.checkbox(
        "Use synthetic fallback dataset",
        value=bool(CONFIG.data.force_synthetic),
    )

    backend_options = ["template", "ollama", "openai"]
    backend_default = str(CONFIG.ui.default_backend)
    backend_index = (
        backend_options.index(backend_default)
        if backend_default in backend_options
        else 0
    )
    backend = st.selectbox(
        "Roadmap backend", backend_options, index=backend_index,
    )

    cf_iterations = st.slider(
        "Counterfactual search iterations",
        min_value=int(CONFIG.ui.cf_iterations_min),
        max_value=int(CONFIG.ui.cf_iterations_max),
        value=int(CONFIG.ui.cf_iterations_default),
    )
    cf_candidates = st.slider(
        "Initial DiCE candidates",
        min_value=int(CONFIG.ui.cf_candidates_min),
        max_value=int(CONFIG.ui.cf_candidates_max),
        value=int(CONFIG.ui.cf_candidates_default),
    )

with st.spinner("Loading model, causal graph, and resource index..."):
    data, predictor, scm, resource_index = load_backend(force_synthetic)

st.success(
    f"Backend ready | data source={data.source} | "
    f"model={predictor.best_model_name}"
)

st.subheader("Dataset case")

n_test = len(data.X_test)
col_a, col_b = st.columns([3, 1])

with col_b:
    denied_only = st.checkbox(
        "Denied cases only",
        value=True,
        help="Show only test cases below the decision threshold.",
    )
    near_boundary_only = st.checkbox(
        "Near-boundary only",
        value=True,
        help=(
            "Show only cases with probability in [0.20, 0.50). "
            "These are the cases where single-step recourse is most "
            "likely to succeed. Uncheck to browse the full denied set."
        ),
    )

# Rebuild the eligible list when the model or filters change.
if (
    "eligible_indices" not in st.session_state
    or st.session_state.get("eligible_model") != predictor.best_model_name
    or st.session_state.get("eligible_denied") != denied_only
    or st.session_state.get("eligible_near") != near_boundary_only
):
    probs = predictor.best_model.predict_proba(data.X_test)[
        :, predictor.positive_class_label
    ]
    threshold = float(CONFIG.predictor.decision_threshold)
    base = []
    for i in range(n_test):
        p = probs[i]
        if denied_only and p >= threshold:
            continue
        if near_boundary_only and not (0.20 <= p < 0.50):
            continue
        base.append(i)
    st.session_state["eligible_indices"] = sorted(
        base, key=lambda i: _priority(i, probs)
    )
    st.session_state["eligible_probs"] = probs
    st.session_state["eligible_model"] = predictor.best_model_name
    st.session_state["eligible_denied"] = denied_only
    st.session_state["eligible_near"] = near_boundary_only

eligible = st.session_state["eligible_indices"]
probs = st.session_state["eligible_probs"]

if not eligible:
    st.warning(
        "No eligible cases match the current filters. "
        "Uncheck 'Denied cases only' or 'Near-boundary only'."
    )
    st.stop()

with col_a:
    case_index = st.selectbox(
        "Select test-set case",
        options=eligible,
        index=0,
        format_func=lambda i: f"Test row {i}  (p = {probs[i]:.1%})",
    )

selected_row = data.X_test.iloc[case_index]
st.markdown("**Selected case profile**")
profile_cols = st.columns(3)
feature_names = list(data.feature_names)
for i, feature in enumerate(feature_names):
    with profile_cols[i % 3]:
        value = selected_row[feature]
        if feature in data.label_encoders:
            enc = data.label_encoders[feature]
            try:
                label = enc.inverse_transform([int(round(value))])[0]
            except Exception:
                label = str(value)
            st.write(f"**{feature.replace('_', ' ')}** — {label}")
        else:
            st.write(f"**{feature.replace('_', ' ')}** — {value:.1f}")

run = st.button("Run pipeline for this case", type="primary")

if run:
    case_row, probability = _predict_case(predictor, data, case_index)

    st.subheader("1. Prediction")
    c1, c2, c3 = st.columns(3)
    c1.metric("Favorable-outcome probability", f"{probability:.1%}")
    c2.metric("Threshold", "0.50")
    c3.metric("Decision", "Favorable" if probability >= 0.5 else "Denied")

    if probability >= float(CONFIG.predictor.decision_threshold):
        st.info(
            "This case is already above the decision threshold; "
            "no recourse search is required."
        )
        st.stop()

    with st.spinner("Generating and validating counterfactuals..."):
        cf_result = generate_counterfactuals(
            predictor,
            scm,
            data,
            query_row=case_row,
            query_probability=probability,
            max_iterations=cf_iterations,
            initial_total_cfs=cf_candidates,
        )

    st.subheader("2. Counterfactual")

    if not cf_result.success or not cf_result.candidates:
        st.warning(
            "No candidate crossed the effective threshold. "
            "The best near-miss is shown below for diagnostics."
        )
        st.write(f"**Rejection breakdown:** {cf_result.rejection_counts}")
        if cf_result.candidates:
            best_near = cf_result.candidates[0]
            st.metric(
                "Best near-miss probability",
                f"{best_near.predicted_probability:.1%}",
                delta=f"{best_near.predicted_probability - probability:+.1%}",
            )
            st.write("**Changes considered:**")
            for feature, (old, new) in best_near.changed_features.items():
                decoded = _decode_change_row(feature, old, new, data)
                st.write(
                    f"- {decoded['Feature']}: "
                    f"{decoded['From']} → {decoded['To']}"
                )
        st.caption(
            "This is the honest output when no valid counterfactual "
            "exists under the current constraints. Try another case, "
            "increase the search budget in the sidebar, or lower the "
            "effective threshold in config.yaml."
        )
        st.stop()

    best = cf_result.candidates[0]

    change_rows = [
        _decode_change_row(feature, old, new, data)
        for feature, (old, new) in best.changed_features.items()
    ]
    st.dataframe(
        pd.DataFrame(change_rows),
        use_container_width=True,
        hide_index=True,
    )
    st.metric(
        "Projected favorable-outcome probability",
        f"{best.predicted_probability:.1%}",
        delta=f"{best.predicted_probability - probability:+.1%}",
    )
    st.caption(
        f"Effective success threshold: "
        f"{cf_result.effective_threshold:.3f} | "
        f"Iterations used: {cf_result.iterations_used}"
    )

    with st.spinner("Retrieving grounded resources..."):
        onet_key = os.environ.get("ONET_API_KEY")
        onet_client = get_onet_client(onet_key)
        rag_result = retrieve_resources_for_candidate(
            best,
            data,
            resource_index,
            onet_client,
            top_k=3,
        )

    st.subheader("3. Evidence retrieval")
    if not rag_result.feature_retrievals:
        st.info(
            "No recommendable features were changed, so no resources "
            "were retrieved. This counterfactual is valid for the "
            "model but not presentable as advice."
        )
    else:
        for fr in rag_result.feature_retrievals:
            with st.expander(fr.feature, expanded=True):
                st.write(fr.query_text)
                for resource in fr.resources:
                    st.markdown(
                        f"- **{resource.title}** — {resource.provider}  \n"
                        f"  Similarity: {resource.similarity_score:.3f}  \n"
                        f"  [Open resource]({resource.url})"
                    )
                if fr.occupation_details:
                    st.write(
                        f"O*NET: {fr.occupation_details.title} "
                        f"({fr.occupation_details.onet_code})"
                    )

    with st.spinner("Generating action plan..."):
        plan = synthesize_action_plan(
            best,
            rag_result,
            data,
            probability,
            backend=backend,
        )

    st.subheader("4. Action plan")
    st.caption(f"Source: {plan.source}")
    for phase in plan.phases:
        with st.expander(
            f"{phase.title} — {phase.timeframe}",
            expanded=True,
        ):
            for action in phase.actions:
                st.write(f"- {action}")

st.divider()
st.caption(
    "Important: counterfactuals describe model behaviour. Retrieved "
    "resources provide evidence for practical exploration; they do not "
    "establish that a resource will cause a specific model feature change "
    "or guarantee an outcome."
)