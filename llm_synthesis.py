"""
llm_synthesis.py
Module 6 of 8: LLM Synthesis (Action Plan Generation)

Design notes
------------
- Only recommendable changes appear in the final plan. If a candidate
  includes a non-recommendable change (marital_status, capital_loss),
  that change is described as a model-side counterfactual but is not
  presented as a suggested action. In practice, rag_retrieval.py
  already filters these out, so template and LLM synthesis see only
  recommendable changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import CONFIG
from counterfactual import CounterfactualCandidate
from loader import DataLoaderResult, RECOMMENDABLE_FEATURES
from rag_retrieval import RAGRetrievalResult


logger = logging.getLogger(__name__)


OLLAMA_HOST_DEFAULT = str(CONFIG.llm.ollama_host)
OLLAMA_MODEL_DEFAULT = str(CONFIG.llm.ollama_model)
OPENAI_MODEL_DEFAULT = str(CONFIG.llm.openai_model)

OLLAMA_TIMEOUT_SECONDS = int(CONFIG.llm.ollama_timeout_seconds)
OLLAMA_NUM_PREDICT = int(CONFIG.llm.ollama_num_predict)
OLLAMA_NUM_CTX = int(CONFIG.llm.ollama_num_ctx)
OPENAI_TIMEOUT_SECONDS = int(CONFIG.llm.openai_timeout_seconds)


@dataclass
class PlanPhase:
    phase_number: int
    title: str
    timeframe: str
    actions: List[str] = field(default_factory=list)


@dataclass
class ActionPlan:
    original_probability: float
    target_probability: float
    phases: List[PlanPhase]
    source: str


def _describe_change(
    feature: str,
    old_value,
    new_value,
    loader_result: DataLoaderResult,
) -> str:
    """Convert a feature change into a readable description."""
    feature_name = str(feature).replace("_", " ")

    if feature == "education_num":
        try:
            old_label = loader_result.education_label_for_years(
                float(old_value)
            )
            new_label = loader_result.education_label_for_years(
                float(new_value)
            )
            if old_label and new_label:
                return (
                    f"years of education: {int(round(float(old_value)))} "
                    f"({old_label}) to "
                    f"{int(round(float(new_value)))} ({new_label})"
                )
        except Exception:
            pass
        return (
            f"years of education: {old_value} to {new_value}"
        )

    try:
        encoders = getattr(loader_result, "label_encoders", {})
        if feature in encoders:
            encoder = encoders[feature]
            old_text = encoder.inverse_transform([int(round(old_value))])[0]
            new_text = encoder.inverse_transform([int(round(new_value))])[0]
            return f"{feature_name}: {old_text} to {new_text}"
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        pass

    return f"{feature_name}: {old_value} to {new_value}"


def _recommendable_changes(
    candidate: CounterfactualCandidate,
) -> dict:
    """Return only the recommendable subset of the candidate's changes."""
    return {
        f: v
        for f, v in candidate.changed_features.items()
        if f in RECOMMENDABLE_FEATURES
    }


def _build_template_plan(
    candidate: CounterfactualCandidate,
    rag_result: RAGRetrievalResult,
    loader_result: DataLoaderResult,
    original_probability: float,
) -> ActionPlan:
    """Build a concise, grounded 3-phase action plan without an LLM."""
    immediate_actions: List[str] = []
    skill_building_actions: List[str] = []

    for fr in rag_result.feature_retrievals:
        if fr.occupation_details:
            occupation_text = (
                f"Research the target role on O*NET: "
                f"{fr.occupation_details.title} "
                f"({fr.occupation_details.onet_code})"
            )
            if fr.occupation_details.bright_outlook:
                occupation_text += (
                    " -- flagged as a Bright Outlook occupation"
                )
            immediate_actions.append(occupation_text)

        for resource in fr.resources:
            action_line = (
                f"{resource.title} "
                f"({resource.provider}) -- {resource.url}"
            )
            if resource.resource_type in (
                "tool", "advisory_service", "job_search",
            ):
                immediate_actions.append(action_line)
            else:
                skill_building_actions.append(action_line)

    immediate_actions = list(dict.fromkeys(immediate_actions))
    skill_building_actions = list(dict.fromkeys(skill_building_actions))

    immediate_actions = immediate_actions[:4]
    skill_building_actions = skill_building_actions[:4]

    recommended = _recommendable_changes(candidate)
    change_summaries = [
        _describe_change(feature, old_value, new_value, loader_result)
        for feature, (old_value, new_value) in recommended.items()
    ]

    if not immediate_actions:
        immediate_actions = [
            "Review the retrieved career and occupation resources.",
            "Research the requirements associated with the target change.",
        ]

    if not skill_building_actions:
        skill_building_actions = [
            "Use the retrieved training or education resources "
            "to work toward the required change."
        ]

    phases = [
        PlanPhase(
            phase_number=1,
            title="Immediate steps",
            timeframe="Weeks 1-2",
            actions=immediate_actions,
        ),
        PlanPhase(
            phase_number=2,
            title="Build the qualifying change",
            timeframe="Months 1-6",
            actions=skill_building_actions,
        ),
        PlanPhase(
            phase_number=3,
            title="Target outcome",
            timeframe="After completing phases 1-2",
            actions=[
                (
                    "Complete the following change(s): "
                    + "; ".join(change_summaries)
                    + "."
                ),
                (
                    "Projected effect: the model estimates that "
                    "the predicted probability of the favorable "
                    f"outcome moves from {original_probability:.1%} "
                    f"today to "
                    f"{candidate.predicted_probability:.1%} "
                    "after the proposed changes."
                ),
                (
                    "This probability is a model estimate, "
                    "not a guarantee of an actual approval "
                    "or outcome."
                ),
            ],
        ),
    ]

    return ActionPlan(
        original_probability=original_probability,
        target_probability=candidate.predicted_probability,
        phases=phases,
        source="template",
    )


def _build_cot_prompt(
    candidate: CounterfactualCandidate,
    rag_result: RAGRetrievalResult,
    loader_result: DataLoaderResult,
    original_probability: float,
) -> str:
    """Build a strongly constrained prompt for Ollama/OpenAI."""
    recommended = _recommendable_changes(candidate)
    change_lines = [
        _describe_change(feature, old, new, loader_result)
        for feature, (old, new) in recommended.items()
    ]

    resource_lines = []
    resource_counter = 1
    for fr in rag_result.feature_retrievals:
        for resource in fr.resources[:2]:
            resource_lines.append(
                f"- [R{resource_counter}] Feature: {fr.feature} | "
                f"Resource: {resource.title} | "
                f"Provider: {resource.provider} | "
                f"Type: {resource.resource_type} | "
                f"Description: {resource.description[:240]} | "
                f"URL: {resource.url}"
            )
            resource_counter += 1

    changes_text = "\n".join(change_lines)

    resources_text = (
        "\n".join(resource_lines)
        if resource_lines
        else "No verified resources were retrieved."
    )

    return f"""
You are generating a concise, evidence-grounded career action plan.

The person received a denied outcome from a predictive model.

Current estimated favorable-outcome probability:
{original_probability:.1%}

Verified counterfactual changes:
{changes_text}

Verified real-world resources retrieved by the system:
{resources_text}

IMPORTANT RULES:

1. Use ONLY the verified resources listed above.
2. When an action depends on a resource, cite its evidence ID
   exactly as [R1], [R2], etc. Do not invent evidence IDs.
3. Do NOT invent organizations, courses, certificates, URLs,
   job requirements, salaries, or other resources.
4. Do NOT invent a new counterfactual change.
5. Do NOT change the feature values given above.
6. Do NOT claim that completing an action guarantees approval.
7. The probability is only a model estimate.
8. Do not fabricate a causal explanation that is not supported
   by the retrieved resources or the counterfactual.
9. Keep the plan practical and concise.
10. Do not repeat the phase headers inside another phase.

STRUCTURE:

Phase 1
Give exactly 2 immediate actions that can reasonably be started
within the first two weeks using the verified resources.

Phase 2
Give exactly 2 practical actions over the next 1-6 months that
are relevant to the verified resources and the counterfactual
change.

Phase 3
Briefly state:
- the verified counterfactual change
- the model's projected probability

Do not invent a different probability.

OUTPUT FORMAT:

Phase 1
- action
- action

Phase 2
- action
- action

Phase 3
- counterfactual change
- projected probability

Respond ONLY with the three sections above.
Do not include an introduction, conclusion, explanation,
or additional headings.
""".strip()


def _clean_action_line(line: str) -> str:
    """Clean common Markdown/list formatting from an LLM action."""
    line = line.strip()
    line = line.replace("**", "")
    line = line.replace("__", "")
    line = re.sub(r"^#{1,6}\s*", "", line)
    line = re.sub(r"^[-*•]\s*", "", line)
    line = re.sub(r"^\d+[\.\)]\s*", "", line)
    return line.strip()


def _is_phase_header(line: str, phase_number: int) -> bool:
    """Recognize Markdown variations of a phase header."""
    cleaned = line.strip().lower()
    cleaned = cleaned.replace("*", "")
    cleaned = cleaned.replace("_", "")
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    return bool(re.match(rf"^phase\s*{phase_number}\b", cleaned))


def _validate_grounding(
    text: str,
    rag_result: RAGRetrievalResult,
) -> bool:
    """Reject responses that introduce URLs/evidence IDs not supplied by RAG."""
    if not text or not text.strip():
        return False

    verified_resources = [
        resource
        for fr in rag_result.feature_retrievals
        for resource in fr.resources
    ]
    verified_urls = {
        resource.url for resource in verified_resources if resource.url
    }
    max_ref = sum(
        min(2, len(fr.resources)) for fr in rag_result.feature_retrievals
    )

    for url in re.findall(r"https?://[^\s)\]]+", text):
        normalized = url.rstrip(".,;")
        if normalized not in verified_urls:
            logger.warning(
                "Grounding check failed: unverified URL %s", normalized
            )
            return False

    for match in re.findall(r"\[R(\d+)\]", text, flags=re.IGNORECASE):
        ref = int(match)
        if ref < 1 or ref > max_ref:
            logger.warning(
                "Grounding check failed: invalid evidence ID [R%s]", match
            )
            return False

    forbidden = (
        "work without pay",
        "work for free",
        "switch to without-pay",
        "move to without-pay",
    )
    lowered = text.lower()
    if any(phrase in lowered for phrase in forbidden):
        logger.warning(
            "Grounding check failed: unsafe/unrealistic workclass wording."
        )
        return False

    return True


def _parse_llm_plan_text(
    text: str,
    candidate: CounterfactualCandidate,
    source: str,
    original_probability: float,
    loader_result: Optional[DataLoaderResult] = None,
) -> Optional[ActionPlan]:
    """Convert the LLM response into an ActionPlan."""
    if not text or not text.strip():
        return None

    text = text.strip()

    recommended = _recommendable_changes(candidate)
    change_summaries = [
        _describe_change(feature, old_value, new_value, loader_result)
        for feature, (old_value, new_value) in recommended.items()
    ]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            phases = []
            for number in (1, 2):
                key = f"phase_{number}"
                phase_data = parsed.get(key)
                if not phase_data:
                    continue
                if isinstance(phase_data, dict):
                    title = phase_data.get(
                        "title",
                        "Immediate steps"
                        if number == 1
                        else "Build the qualifying change",
                    )
                    timeframe = phase_data.get(
                        "timeframe",
                        "Weeks 1-2" if number == 1 else "Months 1-6",
                    )
                    actions = phase_data.get("actions", [])
                    if isinstance(actions, str):
                        actions = [actions]
                    cleaned_actions = [
                        _clean_action_line(str(action))
                        for action in actions
                        if str(action).strip()
                    ]
                    cleaned_actions = cleaned_actions[:2]
                    phases.append(
                        PlanPhase(
                            phase_number=number,
                            title=title,
                            timeframe=timeframe,
                            actions=cleaned_actions,
                        )
                    )

            if phases:
                phases.append(
                    PlanPhase(
                        phase_number=3,
                        title="Target outcome",
                        timeframe="After completing phases 1-2",
                        actions=[
                            (
                                "Counterfactual change: "
                                + "; ".join(change_summaries)
                                + "."
                            ),
                            (
                                "Projected probability: "
                                f"{candidate.predicted_probability:.1%} "
                                "according to the predictive model. "
                                "This is an estimate, not a guarantee."
                            ),
                        ],
                    )
                )
                return ActionPlan(
                    original_probability=original_probability,
                    target_probability=candidate.predicted_probability,
                    phases=phases,
                    source=source,
                )
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    lines = [
        line.strip() for line in text.splitlines() if line.strip()
    ]

    phase_sections = {1: [], 2: [], 3: []}
    current_phase = None

    for line in lines:
        if _is_phase_header(line, 1):
            current_phase = 1
            continue
        if _is_phase_header(line, 2):
            current_phase = 2
            continue
        if _is_phase_header(line, 3):
            current_phase = 3
            continue

        if current_phase is not None:
            cleaned = _clean_action_line(line)
            if not cleaned:
                continue
            lower = cleaned.lower()
            if lower.startswith("immediate steps"):
                continue
            if lower.startswith("build the qualifying change"):
                continue
            if lower.startswith("target outcome"):
                continue
            phase_sections[current_phase].append(cleaned)

    if not any(phase_sections.values()):
        logger.warning(
            "Ollama response did not contain recognizable Phase headers. "
            "Falling back to template synthesis."
        )
        return None

    phase_sections[1] = phase_sections[1][:2]
    phase_sections[2] = phase_sections[2][:2]

    if not phase_sections[1]:
        phase_sections[1] = [
            "Review the verified resources retrieved for the candidate."
        ]

    if not phase_sections[2]:
        phase_sections[2] = [
            "Use the verified resources to work toward the relevant change."
        ]

    phase_sections[3] = [
        (
            "Counterfactual change: "
            + "; ".join(change_summaries)
            + "."
        ),
        (
            "Projected probability: "
            f"{candidate.predicted_probability:.1%} "
            "according to the predictive model. "
            "This is an estimate, not a guarantee."
        ),
    ]

    phases = [
        PlanPhase(
            phase_number=1,
            title="Immediate steps",
            timeframe="Weeks 1-2",
            actions=phase_sections[1],
        ),
        PlanPhase(
            phase_number=2,
            title="Build the qualifying change",
            timeframe="Months 1-6",
            actions=phase_sections[2],
        ),
        PlanPhase(
            phase_number=3,
            title="Target outcome",
            timeframe="After completing phases 1-2",
            actions=phase_sections[3],
        ),
    ]

    return ActionPlan(
        original_probability=original_probability,
        target_probability=candidate.predicted_probability,
        phases=phases,
        source=source,
    )


def _try_ollama_synthesis(
    candidate: CounterfactualCandidate,
    rag_result: RAGRetrievalResult,
    loader_result: DataLoaderResult,
    original_probability: float,
    model: str = OLLAMA_MODEL_DEFAULT,
    host: str = OLLAMA_HOST_DEFAULT,
) -> Optional[ActionPlan]:
    """Attempt synthesis through locally hosted Ollama."""
    prompt = _build_cot_prompt(
        candidate, rag_result, loader_result, original_probability
    )

    try:
        response = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "num_ctx": OLLAMA_NUM_CTX,
                    "seed": 42,
                },
                "keep_alive": "5m",
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        text = response.json().get("response", "")

        if not text.strip():
            logger.warning("Ollama returned an empty response.")
            return None

        if not _validate_grounding(text, rag_result):
            logger.warning(
                "Ollama response failed the grounding validator; "
                "falling back to deterministic template synthesis."
            )
            return None

        return _parse_llm_plan_text(
            text,
            candidate,
            source="ollama",
            original_probability=original_probability,
            loader_result=loader_result,
        )

    except requests.exceptions.RequestException as exc:
        logger.warning(
            "Ollama synthesis failed (%s). "
            "Is Ollama running at %s with '%s' pulled? "
            "Falling back to the template backend.",
            exc, host, model,
        )
        return None


def _try_openai_synthesis(
    candidate: CounterfactualCandidate,
    rag_result: RAGRetrievalResult,
    loader_result: DataLoaderResult,
    original_probability: float,
    model: str = OPENAI_MODEL_DEFAULT,
) -> Optional[ActionPlan]:
    """Attempt synthesis through the OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set. Falling back to template backend."
        )
        return None

    prompt = _build_cot_prompt(
        candidate, rag_result, loader_result, original_probability
    )

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 300,
            },
            timeout=OPENAI_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not text.strip():
            logger.warning("OpenAI returned an empty response.")
            return None

        if not _validate_grounding(text, rag_result):
            logger.warning(
                "OpenAI response failed the grounding validator."
            )
            return None

        return _parse_llm_plan_text(
            text,
            candidate,
            source="openai",
            original_probability=original_probability,
            loader_result=loader_result,
        )

    except requests.exceptions.RequestException as exc:
        logger.warning("OpenAI synthesis failed: %s", exc)
        return None


def synthesize_action_plan(
    candidate: CounterfactualCandidate,
    rag_result: RAGRetrievalResult,
    loader_result: DataLoaderResult,
    original_probability: float,
    backend: str = "template",
    ollama_model: str = OLLAMA_MODEL_DEFAULT,
    ollama_host: str = OLLAMA_HOST_DEFAULT,
    openai_model: str = OPENAI_MODEL_DEFAULT,
) -> ActionPlan:
    """Generate the final action plan."""
    backend = backend.lower().strip()

    if backend == "ollama":
        print(
            f"--- Requesting backend='ollama' "
            f"with model '{ollama_model}' ---"
        )
        result = _try_ollama_synthesis(
            candidate=candidate,
            rag_result=rag_result,
            loader_result=loader_result,
            original_probability=original_probability,
            model=ollama_model,
            host=ollama_host,
        )
        if result is not None:
            return result

    elif backend == "openai":
        print(
            f"--- Requesting backend='openai' "
            f"with model '{openai_model}' ---"
        )
        result = _try_openai_synthesis(
            candidate=candidate,
            rag_result=rag_result,
            loader_result=loader_result,
            original_probability=original_probability,
            model=openai_model,
        )
        if result is not None:
            return result

    elif backend != "template":
        logger.warning(
            "Unknown backend '%s'. Using template backend.", backend
        )

    return _build_template_plan(
        candidate=candidate,
        rag_result=rag_result,
        loader_result=loader_result,
        original_probability=original_probability,
    )


def print_action_plan(plan: ActionPlan) -> None:
    """Print the final action plan."""
    print(f"Source: {plan.source}")
    for phase in plan.phases:
        print()
        print(f"{phase.title} ({phase.timeframe})")
        for action in phase.actions:
            print(f"  - {action}")