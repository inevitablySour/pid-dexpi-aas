"""P&ID -> DEXPI converter GUI for chemists (no CLI needed).

Run with:
    .venv/bin/streamlit run app.py

Upload a .drawio file, see the validate-by-counting report, view the re-rendered P&ID,
and download the DEXPI XML + SVG. Wraps the same pipeline run_pipeline.py uses.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from pid_dexpi_aas.build_diagram import build_diagram
from pid_dexpi_aas.build_model import build_model
from pid_dexpi_aas.decode_drawio import decode_drawio
from pid_dexpi_aas.parser_drawio import parse_drawio_xml_resolved
from pid_dexpi_aas.render_svg import render_diagram_svg
from pid_dexpi_aas.serialize import save_and_verify
from pid_dexpi_aas.validate_by_counting import report_for_diagram

st.set_page_config(page_title="P&ID to DEXPI converter", layout="wide")
st.title("P&ID to DEXPI converter")
st.caption(
    "Upload a draw.io P&ID (.drawio), get a DEXPI XML file, a re-rendered picture to "
    "check against your original drawing, and a report of what was and wasn't converted."
)

uploaded_file = st.file_uploader("Upload a .drawio file", type=["drawio", "xml"])

if uploaded_file is None:
    st.info("Choose a .drawio file to get started.")
    st.stop()

with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_dir_path = Path(tmp_dir)
    input_path = tmp_dir_path / uploaded_file.name
    input_path.write_bytes(uploaded_file.getvalue())

    try:
        xml_variants = decode_drawio(input_path)
        xml_text = xml_variants[0]
    except Exception as exc:
        st.error(f"Could not read this file as a draw.io diagram: {exc}")
        st.stop()

    diagram = parse_drawio_xml_resolved(xml_text, source_file=uploaded_file.name)
    result = build_model(diagram)
    report = report_for_diagram(diagram, result)

    output_stem = Path(uploaded_file.name).stem
    xml_output_path = tmp_dir_path / f"{output_stem}.xml"
    svg_output_path = tmp_dir_path / f"{output_stem}.svg"

    reloaded = save_and_verify(result.model, tmp_dir_path, output_stem)
    diagram_obj = build_diagram(diagram, result, result.model)
    render_diagram_svg(diagram_obj, svg_output_path)

    col_summary, col_downloads = st.columns([2, 1])

    with col_summary:
        st.subheader("Conversion summary")
        st.metric("Shapes + connectors in drawing", report.total_elements)
        m1, m2 = st.columns(2)
        m1.metric("Converted", report.built_count)
        m2.metric("Skipped", report.skipped_count)

        if report.fully_accounted_for:
            st.success("Every shape and connector was accounted for (converted or skipped with a reason).")
        else:
            st.error(
                "Something in the pipeline lost track of an element - this is a bug, "
                "please report it rather than trusting this output."
            )

        explicit_count = report.connections_by_confidence.get("explicit", 0)
        inferred_count = report.inferred_connection_count
        if inferred_count > 0:
            st.warning(
                f"Of {explicit_count + inferred_count} piping connections converted, "
                f"only {explicit_count} were stated directly in your drawing file. The "
                f"other {inferred_count} were GUESSED by finding the nearest shape to a "
                "loose line-end, because the line wasn't snapped exactly onto the shape "
                "when it was drawn. These guesses can be wrong - please check them "
                "against your original drawing before trusting the output. To fix at "
                "the source: in draw.io, drag each connector's loose end until it "
                "snaps onto the shape (you'll see a green highlight)."
            )
        else:
            st.success("Every converted connection was stated directly in your drawing file.")

        if report.multi_tag_shapes:
            st.warning(
                f"{len(report.multi_tag_shapes)} shape(s) picked up more than one nearby "
                "label - check these in the picture below, the label text run together "
                "usually means a stream/state label got attached to the wrong equipment."
            )

    with col_downloads:
        st.subheader("Download")
        st.download_button(
            "Download DEXPI XML",
            data=xml_output_path.read_bytes(),
            file_name=xml_output_path.name,
            mime="application/xml",
        )
        st.download_button(
            "Download re-rendered SVG",
            data=svg_output_path.read_bytes(),
            file_name=svg_output_path.name,
            mime="image/svg+xml",
        )

    st.subheader("Re-rendered P&ID")
    st.caption(
        "Generic ISO-10628-style symbols, not your original drawing style - use this to "
        "check equipment, tags, and connections match your drawing, not for exact looks."
    )
    st.image(str(svg_output_path))

    if report.skipped_details:
        with st.expander(f"Skipped elements ({len(report.skipped_details)})"):
            for detail in report.skipped_details:
                st.text(detail)
