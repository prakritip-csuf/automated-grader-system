from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("GRADER_API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024

DEFAULT_RUBRIC = pd.DataFrame(
    [
        {
            "id": "photosynthesis",
            "description": "Identifies photosynthesis",
            "points": 2.0,
            "references": "Plants use sunlight for photosynthesis. || Photosynthesis requires light energy.",
        },
        {
            "id": "energy",
            "description": "Explains sunlight as an energy source",
            "points": 2.0,
            "references": "Sunlight provides energy for plants. || Plants use light energy to make food.",
        },
        {
            "id": "food",
            "description": "Mentions food or glucose production",
            "points": 1.0,
            "references": "Plants produce glucose. || Plants make their own food.",
        },
    ]
)


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "answer_text": "",
        "raw_text": "",
        "extraction_details": None,
        "upload_digest": None,
        "grade_bundle": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_answer() -> None:
    """Reset text and grading state when the input source changes."""
    st.session_state.answer_text = ""
    st.session_state.raw_text = ""
    st.session_state.extraction_details = None
    st.session_state.upload_digest = None
    st.session_state.grade_bundle = None


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_rubric(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert editable rubric rows to the exact FastAPI GradeRequest schema."""
    criteria: list[dict[str, Any]] = []
    ids: set[str] = set()

    for row_index, row in table.iterrows():
        def cell(name: str) -> str:
            value = row[name]
            return "" if pd.isna(value) else str(value).strip()

        criterion_id = cell("id")
        description = cell("description")
        raw_references = cell("references")
        points_raw = row["points"]
        points_missing = pd.isna(points_raw)

        # Ignore an entirely blank row created by the dynamic table editor.
        if not criterion_id and not description and not raw_references and points_missing:
            continue

        if not criterion_id or not description or not raw_references or points_missing:
            raise ValueError(f"Rubric row {row_index + 1}: complete every field.")

        try:
            points = float(points_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Rubric row {row_index + 1}: points must be numeric.") from exc

        if not 0 < points <= 100:
            raise ValueError(f"Rubric row {row_index + 1}: points must be between 0 and 100.")

        if criterion_id in ids:
            raise ValueError(f"Duplicate rubric criterion ID: {criterion_id}")
        ids.add(criterion_id)

        references = [part.strip() for part in raw_references.split("||") if part.strip()]
        if not references:
            raise ValueError(f"Rubric row {row_index + 1}: add at least one reference.")

        criteria.append(
            {
                "id": criterion_id,
                "description": description,
                "points": points,
                "references": references,
            }
        )

    if not criteria:
        raise ValueError("Add at least one rubric criterion.")

    return criteria


def api_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/api/v1{path}"


def raise_api_error(response: requests.Response) -> None:
    """Convert FastAPI error responses into readable UI messages."""
    try:
        payload = response.json()
        detail = payload.get("detail", response.text)
    except ValueError:
        detail = response.text
    raise RuntimeError(f"API returned HTTP {response.status_code}: {str(detail)[:900]}")


def post_json(base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.post(api_url(base, path), json=payload, timeout=240)
    except requests.RequestException as exc:
        raise RuntimeError(f"Cannot reach the grading backend: {exc}") from exc
    if not response.ok:
        raise_api_error(response)
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("The backend returned an invalid JSON response.") from exc


def post_upload(
    base: str,
    paths: list[str],
    filename: str,
    data: bytes,
    content_type: str,
) -> tuple[dict[str, Any], str]:
    """Upload once per path; retry only for a missing (404) route."""
    for index, path in enumerate(paths):
        try:
            response = requests.post(
                api_url(base, path),
                files={"file": (filename, io.BytesIO(data), content_type)},
                timeout=300,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Cannot reach the extraction backend: {exc}") from exc

        if response.status_code == 404 and index < len(paths) - 1:
            continue
        if not response.ok:
            raise_api_error(response)
        try:
            return response.json(), path
        except ValueError as exc:
            raise RuntimeError("The backend returned an invalid JSON response.") from exc

    raise RuntimeError("No matching upload endpoint was found.")


def display_result(result: dict[str, Any]) -> pd.DataFrame:
    """Display scores, review flags, and per-criterion evidence."""
    score = float(result.get("score", 0))
    maximum = float(result.get("max_score", 0))
    similarity = float(result.get("overall_similarity", 0))

    left, middle, right = st.columns(3)
    left.metric("Predicted score", f"{score:.2f} / {maximum:g}")
    middle.metric("Rubric points earned", f"{100 * score / maximum:.1f}%" if maximum else "N/A")
    right.metric("Overall cosine similarity", f"{similarity:.3f}")
    st.caption("Cosine similarity is not a probability of correctness or calibrated confidence.")

    if result.get("review_required", False):
        st.warning("Instructor review recommended.")
        for reason in result.get("review_reasons", []):
            st.write(f"• {reason}")
    else:
        st.success("No rule-based review flags were triggered. Instructor verification is still advised.")

    rows: list[dict[str, Any]] = []
    for item in result.get("criteria", []):
        rows.append(
            {
                "Criterion": item.get("description", ""),
                "Awarded": item.get("awarded_points", 0),
                "Available": item.get("available_points", 0),
                "Similarity": item.get("similarity", 0),
                "Student evidence": item.get("matched_student_unit", ""),
                "Matched reference": item.get("matched_reference", ""),
                "Contradiction probability": item.get("contradiction_probability"),
                "Review": item.get("review", False),
            }
        )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        st.info("No criterion-level matches were returned (for example, the answer may be blank).")
    else:
        st.subheader("Criterion-by-criterion scoring")
        st.dataframe(dataframe, hide_index=True, width="stretch")
        with st.expander("How the score was calculated"):
            st.write(
                "Each rubric criterion is matched to the most similar student sentence. "
                "The configured partial/full-credit thresholds convert similarity to marks. "
                "Review flags are heuristic and the thresholds require validation against human scores."
            )
    return dataframe


st.set_page_config(
    page_title="Automated Answer Grader",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)
initialize_state()

# Sidebar: connection and controls.
with st.sidebar:
    st.header("Demo controls")
    base_url = st.text_input("FastAPI server", value=DEFAULT_API_URL).rstrip("/")
    st.caption("The backend runs TrOCR, PDF extraction, and semantic grading.")
    if st.button("Check backend connection", width="stretch"):
        try:
            health = requests.get(api_url(base_url, "/health"), timeout=5)
            if health.ok:
                st.success("Backend is reachable")
                st.json(health.json())
            else:
                st.error(f"Health endpoint returned HTTP {health.status_code}")
        except requests.RequestException as exc:
            st.error(f"Backend is unavailable: {exc}")
    if st.button("Clear student answer and results", width="stretch"):
        reset_answer()
        st.rerun()
    st.divider()
    st.markdown("**Workflow**")
    st.write("1. Define the question and rubric")
    st.write("2. Upload or type an answer")
    st.write("3. Verify/correct extracted text")
    st.write("4. Grade and inspect evidence")
    st.write("5. Review and export")
    st.caption("Local demonstration only. Avoid uploading identifiable student records.")

st.title("Automated Short-Answer Grader using TrOCR and NLP")
st.info(
    "Instructor-assistance prototype. Review extracted text and predicted marks before using them. "
    "This UI does not persist student records or instructor overrides."
)


st.header("1. Question and marking rubric")
question = st.text_input("Question", value="Why do plants need sunlight?")
reference_answer = st.text_area(
    "Instructor reference answer",
    value="Plants use sunlight during photosynthesis to convert light energy into chemical energy and produce glucose.",
    height=95,
)
st.caption("Edit the rubric below. Separate alternative acceptable statements using || (two vertical bars).")
rubric_table = st.data_editor(
    DEFAULT_RUBRIC,
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    key="rubric_table",
    column_config={
        "id": st.column_config.TextColumn("Criterion ID", required=True),
        "description": st.column_config.TextColumn("Required concept", required=True, width="medium"),
        "points": st.column_config.NumberColumn("Max points", min_value=0.5, step=0.5, required=True),
        "references": st.column_config.TextColumn("Acceptable statements (separate with ||)", required=True, width="large"),
    },
)
try:
    criteria = parse_rubric(rubric_table)
    rubric_error = None
    st.caption(f"{len(criteria)} criteria · {sum(c['points'] for c in criteria):g} maximum points")
except ValueError as exc:
    criteria = []
    rubric_error = str(exc)
    st.error(f"Rubric needs attention: {rubric_error}")

# Section 2: Source selection and extraction.
st.header("2. Student response")
mode = st.radio(
    "Choose the input type",
    ["Handwritten image", "PDF document", "Typed answer"],
    horizontal=True,
    key="input_mode",
    on_change=reset_answer,
)

if mode in {"Handwritten image", "PDF document"}:
    image_mode = mode == "Handwritten image"
    upload = st.file_uploader(
        "Upload a cropped handwritten answer" if image_mode else "Upload an answer-only PDF",
        type=["png", "jpg", "jpeg"] if image_mode else ["pdf"],
        key="image_upload" if image_mode else "pdf_upload",
    )
    if upload is not None:
        content = upload.getvalue()
        new_digest = hashlib.sha256(content).hexdigest()
        if new_digest != st.session_state.upload_digest:
            # Prevent old OCR text from being graded against a new file.
            st.session_state.answer_text = ""
            st.session_state.raw_text = ""
            st.session_state.extraction_details = None
            st.session_state.grade_bundle = None
            st.session_state.upload_digest = new_digest

        st.caption(f"File: {upload.name} · {len(content) / 1024:.1f} KB")
        if image_mode:
            st.image(upload, caption="Uploaded handwriting", width="stretch")
        else:
            st.caption("PDF extraction uses embedded text when available and TrOCR for image-only pages.")

        max_size = MAX_IMAGE_BYTES if image_mode else MAX_PDF_BYTES
        if len(content) > max_size:
            st.error(f"This file exceeds the {max_size // (1024 * 1024)} MB demo limit.")
        elif st.button("Recognize handwriting" if image_mode else "Extract PDF text", type="primary"):
            paths = ["/handwriting", "/recognition/handwriting"] if image_mode else ["/extraction/pdf"]
            content_type = upload.type or ("image/jpeg" if image_mode else "application/pdf")
            try:
                with st.spinner("Recognizing handwriting..." if image_mode else "Extracting PDF text..."):
                    response, used_path = post_upload(
                        base_url, paths, upload.name, content, content_type
                    )
                extracted = str(response.get("recognized_text", response.get("text", "")))
                st.session_state.raw_text = extracted
                st.session_state.answer_text = extracted
                st.session_state.extraction_details = {
                    "filename": upload.name,
                    "endpoint": used_path,
                    "method": response.get("method", "trocr_ocr" if image_mode else "unknown"),
                    "lines": response.get("lines", []),
                    "line_count": response.get("line_count"),
                }
                st.session_state.grade_bundle = None
                if extracted.strip():
                    st.success("Extraction completed. Review the text below before grading.")
                else:
                    st.warning("No text was recognized. Try a clearer, tightly cropped answer.")
            except (RuntimeError, ValueError) as exc:
                st.error(str(exc))

if st.session_state.extraction_details is not None:
    details = st.session_state.extraction_details
    st.caption(f"Extraction method: {details['method']} · Route: {details['endpoint']}")
    if details.get("line_count") is not None:
        st.caption(f"Detected line crops: {details['line_count']} (not a measure of OCR accuracy)")
    with st.expander("View raw extracted text and individual lines"):
        st.code(st.session_state.raw_text or "(empty)", language=None)
        if details.get("lines"):
            for number, line in enumerate(details["lines"], start=1):
                st.write(f"Line {number}: {line}")

st.subheader("3. Verify or edit the student answer")
st.text_area(
    "Student answer used for grading",
    key="answer_text",
    height=160,
    placeholder="Type an answer or extract text from an uploaded image/PDF.",
)
st.caption("Line breaks are normalized by the NLP backend. Keep negations and technical terms intact.")


st.header("4. Semantic grading")
compare_raw = st.checkbox(
    "Also grade the raw OCR text for comparison (when edited)",
    value=True,
    help="A single-answer demonstration of OCR error propagation, not a dataset-level evaluation.",
)

payload = {
    "question": question.strip(),
    "reference_answer": reference_answer.strip(),
    "student_answer": st.session_state.answer_text,
    "criteria": criteria,
}
can_grade = bool(payload["question"] and payload["reference_answer"] and payload["student_answer"].strip() and criteria)
signature = fingerprint({"api_base": base_url, "payload": payload, "raw": st.session_state.raw_text, "compare": compare_raw})

if st.button("Grade student answer", type="primary", disabled=not can_grade):
    try:
        with st.spinner("Calculating rubric-based semantic scores..."):
            graded = post_json(base_url, "/grading/grade", payload)
            raw_result = None
            raw_text = st.session_state.raw_text
            if compare_raw and raw_text.strip() and raw_text != payload["student_answer"]:
                raw_payload = {**payload, "student_answer": raw_text}
                raw_result = post_json(base_url, "/grading/grade", raw_payload)

        st.session_state.grade_bundle = {
            "signature": signature,
            "graded": graded,
            "raw_graded": raw_result,
        }
        st.success("Grading completed.")
    except (RuntimeError, ValueError) as exc:
        st.session_state.grade_bundle = None
        st.error(str(exc))

bundle = st.session_state.grade_bundle
if bundle is not None and bundle["signature"] != signature:
    st.info("The input or rubric has changed. Click 'Grade student answer' to update the results.")
elif bundle is not None:
    result = bundle["graded"]
    criterion_frame = display_result(result)

    if bundle["raw_graded"] is not None:
        raw_score = float(bundle["raw_graded"].get("score", 0))
        corrected_score = float(result.get("score", 0))
        st.subheader("Raw OCR versus corrected-text grading")
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw OCR score", f"{raw_score:.2f}")
        c2.metric("Edited text score", f"{corrected_score:.2f}")
        c3.metric("Score difference", f"{corrected_score - raw_score:+.2f}")
        st.caption("These are model predictions, not human-assigned grades. A difference does not establish which grade is correct.")

    
    st.header("5. Instructor review and export")
    review_key = signature[:12]
    override = st.checkbox("Record an instructor-adjusted score in the export", key=f"override_{review_key}")
    reviewed_score = None
    if override:
        reviewed_score = st.number_input(
            "Instructor-adjusted score",
            min_value=0.0,
            max_value=float(result["max_score"]),
            value=float(result["score"]),
            step=0.5,
            key=f"reviewed_score_{review_key}",
        )
    review_note = st.text_area(
        "Instructor review note (optional)",
        key=f"review_note_{review_key}",
        placeholder="Explain any manual correction or score adjustment.",
    )
    st.caption("Overrides and notes are not saved in the backend. They are included only in the downloaded JSON.")

    export = {
        "question": payload["question"],
        "reference_answer": payload["reference_answer"],
        "rubric": payload["criteria"],
        "input_type": mode,
        "extraction": st.session_state.extraction_details,
        "raw_extracted_text": st.session_state.raw_text or None,
        "verified_student_answer": payload["student_answer"],
        "model_result": result,
        "raw_ocr_model_result": bundle["raw_graded"],
        "instructor_review": {
            "overridden": override,
            "reviewed_score": float(reviewed_score) if reviewed_score is not None else None,
            "note": review_note,
        },
    }
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download grading report (JSON)",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name="automated_grading_report.json",
        mime="application/json",
        width="stretch",
    )
    if not criterion_frame.empty:
        d2.download_button(
            "Download rubric scores (CSV)",
            data=criterion_frame.to_csv(index=False),
            file_name="rubric_scores.csv",
            mime="text/csv",
            width="stretch",
        )

# Optional research dashboard: only display saved, real evaluation outputs.
with st.expander("Research evaluation summaries (optional)"):
    found_results = False

    ocr_dir = Path("artifacts/ocr")
    for summary_file in sorted(ocr_dir.glob("*_summary.csv")) if ocr_dir.exists() else []:
        try:
            st.markdown(f"**TrOCR evaluation: {summary_file.stem}**")
            st.dataframe(pd.read_csv(summary_file), hide_index=True, width="stretch")
            found_results = True
        except (OSError, pd.errors.ParserError) as exc:
            st.warning(f"Could not read {summary_file.name}: {exc}")

    comparison_file = Path("artifacts/model_comparison/comparison.csv")
    if comparison_file.is_file():
        try:
            st.markdown("**Semantic model comparison**")
            st.dataframe(pd.read_csv(comparison_file), hide_index=True, width="stretch")
            found_results = True
        except (OSError, pd.errors.ParserError) as exc:
            st.warning(f"Could not read model comparison: {exc}")

    end_to_end_file = Path("artifacts/end_to_end/summary.json")
    if end_to_end_file.is_file():
        try:
            st.markdown("**End-to-end OCR-to-grading evaluation**")
            st.json(json.loads(end_to_end_file.read_text(encoding="utf-8")))
            found_results = True
        except (OSError, ValueError) as exc:
            st.warning(f"Could not read end-to-end results: {exc}")

    if not found_results:
        st.info("No saved evaluation summaries found yet. Run the offline evaluation scripts first.")
    st.caption("These are saved offline experiment results, not live accuracy estimates for the current answer.")

st.divider()
st.caption(
    "Research prototype limitations: scanned PDF OCR assumes answer-oriented pages; "
    "semantic similarity does not guarantee factual correctness; "
    "review flags are heuristic; no persistent grading database is implemented."
)