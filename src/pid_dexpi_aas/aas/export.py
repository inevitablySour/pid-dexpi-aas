"""DEXPI -> Asset Administration Shell: an AASX package with the IDTA 02012-1-0 Submodel
"Information Model for P&I Diagrams based on DEXPI Standard" (Type 1, a static container).

The template wraps DEXPI files rather than re-modelling them:

    Submodel DEXPI
      PlantMetadata (SMC)        enterprise / site / plant / section / project ...
      Model01, Model02 ... (SMC) one per DEXPI file
        ModelMetadata (SMC)      drawing number, creator, dates ...
        ModelFile (File)         the DEXPI ProteusXML, inside the package
        ModelRepresentation      the drawing as SVG (optional)
        MappingDirectory (SMC)   one SMC per tagged element: TagName / SubTagName, Class,
                                 LocalId, and a RelationshipElement from the element inside
                                 the file (FragmentReference "ProteusXML@ID=<id>") to a stable,
                                 global asset id - what the digital twins look elements up by.

The MappingDirectory is read from the exported XML itself, so its local ids are exactly the
file's. Tagged elements per the template: equipment (TagNameAssignmentClass), process
instrumentation functions (category + number), nozzles as sub tags. Valves are added the
same way from their piping component name (the template's TagMapping does not name them,
but a valve tag is just as much an engineering identifier the twins need).

Uses Eclipse BaSyx (basyx-python-sdk); check_aasx() runs the IDTA's aas-test-engines.
"""
from __future__ import annotations

import datetime as dt
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from basyx.aas import model
from basyx.aas.adapter import aasx

SEM = "https://admin-shell.io/idta/DEXPI/1/0/"
RDL = "http://sandbox.dexpi.org/rdl/"

# PlantMetadata (Table 2) and ModelMetadata (Table 4): idShort -> semantic id
PLANT_FIELDS = {
    "EnterpriseIdentificationCode": RDL + "EnterpriseIdentificationCodeAssignmentClass",
    "EnterpriseName": RDL + "EnterpriseNameAssignmentClass",
    "SiteIdentificationCode": RDL + "SiteIdentificationCodeAssignmentClass",
    "SiteName": RDL + "SiteNameAssignmentClass",
    "IndustrialComplexIdentificationCode": RDL + "IndustrialComplexIdentificationCodeAssignmentClass",
    "IndustrialComplexName": RDL + "IndustrialComplexNameAssignmentClass",
    "ProcessPlantIdentificationCode": RDL + "ProcessPlantIdentificationCodeAssignmentClass",
    "ProcessPlantName": RDL + "ProcessPlantNameAssignmentClass",
    "PlantSectionIdentificationCode": RDL + "PlantSectionIdentificationCodeAssignmentClass",
    "PlantSectionName": RDL + "PlantSectionNameAssignmentClass",
    "ProjectName": RDL + "ProjectNameAssignmentClass",
    "ProjectNumber": RDL + "ProjectNumberAssignmentClass",
    "SubProjectName": RDL + "SubProjectNameAssignmentClass",
    "SubProjectNumber": RDL + "SubProjectNumberAssignmentClass",
    "ManufacturerName": RDL + "ManufacturerNameAssignmentClass",
}
MODEL_FIELDS = {
    "ApprovalDate": RDL + "ApprovalDateRepresentationAssignmentClass",
    "ApproverName": RDL + "ApproverNameAssignmentClass",
    "ArchiveNumber": RDL + "ArchiveNumberAssignmentClass",
    "CheckerName": RDL + "CheckerNameAssignmentClass",
    "CreationDate": RDL + "CreationDateRepresentationAssignmentClass",
    "CreatorName": RDL + "CreatorNameAssignmentClass",
    "DesignerName": RDL + "DesignerNameAssignmentClass",
    "DrawingNumber": RDL + "DrawingNumberAssignmentClass",
    "DrawingSubTitle": RDL + "DrawingSubTitleAssignmentClass",
    "LastModificationDate": RDL + "LastModificationDateRepresentationAssignmentClass",
}
_DATE_FIELDS = {"ApprovalDate", "CreationDate", "LastModificationDate"}
_MLP_FIELDS = {"DrawingSubTitle"}


@dataclass
class DexpiModelInput:
    name: str  # file stem, e.g. "P&ID gramflow reactor V3"
    dexpi_xml: bytes
    svg: bytes | None = None
    metadata: dict = field(default_factory=dict)  # MODEL_FIELDS idShort -> value


@dataclass
class MappingEntry:
    local_id: str
    cls: str
    tag: str  # TagName, or SubTagName for a sub tag
    parent: str | None = None  # sub tags: the parent element's local id
    kind: str = "TagMapping"  # TagMapping | SubTagMapping


def _ref(iri: str) -> model.ExternalReference:
    return model.ExternalReference((model.Key(model.KeyTypes.GLOBAL_REFERENCE, iri),))


def _id_short(text: str) -> str:
    """AAS idShort: letters, digits, '_' only, starting with a letter."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", text)
    return s if s[:1].isalpha() else f"X_{s}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "plant"


def mapping_entries(dexpi_xml: bytes) -> list[MappingEntry]:
    """Tagged elements of a ProteusXML DEXPI file, with their local ids."""
    root = ET.fromstring(dexpi_xml)
    out = []

    def attrs(el):
        return {g.get("Name"): g.get("Value") for g in el.findall("GenericAttributes/GenericAttribute")}

    def walk(el, parent_id=None):
        eid = el.get("ID")
        a = attrs(el) if eid else {}
        cls = el.get("ComponentClass") or el.tag
        if eid:
            if a.get("TagNameAssignmentClass"):
                out.append(MappingEntry(eid, cls, a["TagNameAssignmentClass"]))
            elif el.tag == "ProcessInstrumentationFunction" and a.get(
                    "ProcessInstrumentationFunctionNumberAssignmentClass"):
                tag = " ".join(x for x in (a.get("ProcessInstrumentationFunctionCategoryAssignmentClass"),
                                           a["ProcessInstrumentationFunctionNumberAssignmentClass"]) if x)
                out.append(MappingEntry(eid, "ProcessInstrumentationFunction", tag))
            elif a.get("PipingComponentNameAssignmentClass"):
                out.append(MappingEntry(eid, cls, a["PipingComponentNameAssignmentClass"]))
            elif a.get("TagName"):  # pid-dexpi-aas custom attribute (classes without a name field)
                out.append(MappingEntry(eid, cls, a["TagName"]))
            elif a.get("SubTagNameAssignmentClass") and parent_id:
                out.append(MappingEntry(eid, cls, a["SubTagNameAssignmentClass"], parent_id, "SubTagMapping"))
        for child in el:
            walk(child, eid if (eid and el.tag in ("Equipment", "PipingComponent")) else parent_id)

    walk(root)
    return out


def _prop(id_short: str, value, semantic: str, value_type=model.datatypes.String) -> model.Property:
    return model.Property(id_short, value_type, value=value, semantic_id=_ref(semantic))


def _metadata_smc(id_short: str, semantic: str, fields: dict, values: dict) -> model.SubmodelElementCollection:
    elements = []
    for name, sem in fields.items():
        v = values.get(name)
        if v in (None, ""):
            continue
        if name in _MLP_FIELDS:
            elements.append(model.MultiLanguageProperty(name, value=model.MultiLanguageTextType({"en": str(v)}),
                                                        semantic_id=_ref(sem)))
        elif name in _DATE_FIELDS and isinstance(v, (dt.date, str)) and _as_date(v):
            elements.append(_prop(name, _as_date(v), sem, model.datatypes.Date))
        else:
            elements.append(_prop(name, str(v), sem))
    return model.SubmodelElementCollection(id_short, elements, semantic_id=_ref(semantic))


def _as_date(v):
    if isinstance(v, dt.date):
        return model.datatypes.Date(v.year, v.month, v.day)
    try:
        d = dt.date.fromisoformat(str(v)[:10])
        return model.datatypes.Date(d.year, d.month, d.day)
    except ValueError:
        return None


def build_package(models: list[DexpiModelInput], plant: dict, base_iri: str | None = None,
                  asset_base: str | None = None):
    """(object store, file store, aas id) for one plant (segment) with its DEXPI models."""
    name = plant.get("ProcessPlantName") or plant.get("PlantSectionName") or plant.get("ProjectName") or models[0].name
    base = (base_iri or f"urn:pid-dexpi-aas:{_slug(name)}").rstrip(":/")
    asset_base = (asset_base or f"{base}:asset").rstrip(":/")
    sm_id = f"{base}:submodel:dexpi"
    files = aasx.DictSupplementaryFileContainer()
    model_smcs = []
    for n, m in enumerate(models, start=1):
        smc_id = f"Model{n:02d}"
        # lowercase: the OPC writer lowercases part names in the relationships, the zip
        # keeps their case - a mixed-case name is "missing" to the IDTA checker
        safe = re.sub(r"[^a-z0-9._-]", "_", m.name.lower())
        xml_path = files.add_file(f"/aasx/files/{safe}.xml", io.BytesIO(m.dexpi_xml), "application/xml")
        elements = [_metadata_smc("ModelMetadata", SEM + "ModelMetadata", MODEL_FIELDS,
                                  {"DrawingNumber": m.name, "CreationDate": dt.date.today(),
                                   "LastModificationDate": dt.date.today(), **m.metadata}),
                    model.File("ModelFile", "application/xml", value=xml_path, semantic_id=_ref(SEM + "ModelFile"))]
        if m.svg:
            svg_path = files.add_file(f"/aasx/files/{safe}.svg", io.BytesIO(m.svg), "image/svg+xml")
            elements.append(model.File("ModelRepresentation", "image/svg+xml", value=svg_path,
                                       semantic_id=_ref(SEM + "ModelRepresentation")))
        mapping, seen = [], set()
        file_ref = (model.Key(model.KeyTypes.SUBMODEL, sm_id),
                    model.Key(model.KeyTypes.SUBMODEL_ELEMENT_COLLECTION, smc_id),
                    model.Key(model.KeyTypes.FILE, "ModelFile"))
        for e in mapping_entries(m.dexpi_xml):
            short = _id_short(e.local_id)
            if short in seen:
                continue
            seen.add(short)
            first = model.ModelReference(file_ref + (model.Key(model.KeyTypes.FRAGMENT_REFERENCE,
                                                               f"ProteusXML@ID={e.local_id}"),), model.File)
            # stable global id: the tag within the plant (a nozzle: its parent's tag + sub tag)
            global_tag = e.tag if e.kind == "TagMapping" else f"{_parent_tag(m.dexpi_xml, e.parent)}/{e.tag}"
            second = _ref(f"{asset_base}:{_slug(global_tag)}")
            props = ([_prop("TagName", e.tag, SEM + "TagName")] if e.kind == "TagMapping" else
                     [_prop("SubTagName", e.tag, SEM + "SubTagName"),
                      _prop("ParentLocalId", e.parent, SEM + "ParentLocalId")])
            props += [_prop("Class", e.cls, SEM + "Class"), _prop("LocalId", e.local_id, SEM + "LocalId"),
                      model.RelationshipElement(f"{short}_rel", first, second,
                                                semantic_id=_ref(SEM + "MappingRelationship"))]
            sem = SEM + ("TagMapping" if e.kind == "TagMapping" else "Metadata/SubTagMapping")
            mapping.append(model.SubmodelElementCollection(short, props, semantic_id=_ref(sem)))
        elements.append(model.SubmodelElementCollection("MappingDirectory", mapping,
                                                        semantic_id=_ref(SEM + "MappingDirectory")))
        model_smcs.append(model.SubmodelElementCollection(smc_id, elements, semantic_id=_ref(SEM + "Model")))
    submodel = model.Submodel(
        sm_id, id_short="DEXPI", semantic_id=_ref(SEM + "Submodel"), kind=model.ModellingKind.INSTANCE,
        submodel_element=[_metadata_smc("PlantMetadata", SEM + "PlantMetadata", PLANT_FIELDS, plant)] + model_smcs,
        description=model.MultiLanguageTextType({"en": f"P&ID of {name} as DEXPI (IDTA 02012-1-0)"}))
    aas_id = f"{base}:aas"
    submodels = [submodel]
    hierarchy = _hierarchy(models, f"{base}:submodel:hierarchy", f"{asset_base}:plant", asset_base, name)
    if hierarchy is not None:
        submodels.append(hierarchy)
    shell = model.AssetAdministrationShell(
        model.AssetInformation(model.AssetKind.INSTANCE, global_asset_id=f"{asset_base}:plant"),
        aas_id, id_short=_id_short(_slug(name).replace("-", "_")) or "Plant",
        submodel={model.ModelReference.from_referable(sm) for sm in submodels},
        description=model.MultiLanguageTextType({"en": f"{name} (plant segment)"}))
    return model.DictObjectStore([shell, *submodels]), files, aas_id


HS = "https://admin-shell.io/idta/HierarchicalStructures/"


def _sections(dexpi_xml: bytes) -> dict[str, str]:
    """{local id: plant section} - the PlantSectionName attribute build_model writes on items
    drawn inside a section frame (only when frames are used as plant sections)."""
    out = {}
    for el in ET.fromstring(dexpi_xml).iter():
        for ga in el.iter("GenericAttribute"):
            if ga.get("Name") == "PlantSectionName" and el.get("ID") and ga.get("Value"):
                out.setdefault(el.get("ID"), ga.get("Value"))
    return out


def _hierarchy(models, sm_id: str, plant_asset: str, asset_base: str, name: str):
    """IDTA 02011-1-1 Hierarchical Structures (ArcheType "Full"): the plant (EntryNode) has its
    plant sections, each section its tagged items (HasPart) - across all the drawings. None
    when no item carries a plant section."""
    by_section: dict[str, dict[str, str]] = {}
    for m in models:
        secs = _sections(m.dexpi_xml)
        for e in mapping_entries(m.dexpi_xml):
            if e.kind == "TagMapping" and e.local_id in secs:
                by_section.setdefault(secs[e.local_id], {})[e.tag] = f"{asset_base}:{_slug(e.tag)}"
    if not by_section:
        return None
    entry_short = _id_short(_slug(name).replace("-", "_")) or "Plant"
    ref = lambda *path: model.ModelReference(
        (model.Key(model.KeyTypes.SUBMODEL, sm_id),) + tuple(model.Key(model.KeyTypes.ENTITY, p) for p in path), model.Entity)
    sections, top_rels = [], []
    for n, (sec, items) in enumerate(sorted(by_section.items()), start=1):
        sec_short = _id_short(f"Section_{_slug(sec)}") or f"Section{n:02d}"
        children, rels = [], []
        for tag, asset in sorted(items.items()):
            short = _id_short(f"Item_{_slug(tag)}")
            children.append(model.Entity(short, model.EntityType.SELF_MANAGED_ENTITY, global_asset_id=asset,
                                         semantic_id=_ref(HS + "Node/1/0")))
            rels.append(model.RelationshipElement(f"HasPart_{short}", ref(entry_short, sec_short),
                                                  ref(entry_short, sec_short, short), semantic_id=_ref(HS + "HasPart/1/0")))
        sections.append(model.Entity(sec_short, model.EntityType.CO_MANAGED_ENTITY, statement=children + rels,
                                     semantic_id=_ref(HS + "Node/1/0"),
                                     description=model.MultiLanguageTextType({"en": f"Plant section {sec}"})))
        top_rels.append(model.RelationshipElement(f"HasPart_{sec_short}", ref(entry_short), ref(entry_short, sec_short),
                                                  semantic_id=_ref(HS + "HasPart/1/0")))
    entry = model.Entity(entry_short, model.EntityType.SELF_MANAGED_ENTITY, global_asset_id=plant_asset,
                         statement=sections + top_rels, semantic_id=_ref(HS + "EntryNode/1/0"))
    return model.Submodel(
        sm_id, id_short="HierarchicalStructures", semantic_id=_ref(HS + "1/1/Submodel"), kind=model.ModellingKind.INSTANCE,
        submodel_element=[_prop("ArcheType", "Full", HS + "ArcheType/1/0"), entry],
        description=model.MultiLanguageTextType({"en": f"{name}: its plant sections and their items (IDTA 02011-1-1)"}))


_PARENT_TAGS: dict = {}


def _parent_tag(dexpi_xml: bytes, parent_id: str | None) -> str:
    key = (id(dexpi_xml), parent_id)
    if key not in _PARENT_TAGS:
        tags = {e.local_id: e.tag for e in mapping_entries(dexpi_xml) if e.kind == "TagMapping"}
        for lid, tag in tags.items():
            _PARENT_TAGS[(id(dexpi_xml), lid)] = tag
    return _PARENT_TAGS.get(key) or (parent_id or "")


VERSIONS = ("3.0", "3.1")  # AAS metamodel versions offered
_NS = {"3.0": "https://admin-shell.io/aas/3/0", "3.1": "https://admin-shell.io/aas/3/1"}


def write_aasx(path_or_file, models: list[DexpiModelInput], plant: dict, version: str = "3.0", **kw) -> str:
    """Write the AASX package; returns the AAS id."""
    Path(path_or_file).write_bytes(aasx_bytes(models, plant, version=version, **kw)) if isinstance(
        path_or_file, (str, Path)) else path_or_file.write(aasx_bytes(models, plant, version=version, **kw))
    return build_package(models, plant, **kw)[2]


def aasx_bytes(models: list[DexpiModelInput], plant: dict, version: str = "3.0", **kw) -> bytes:
    """The AASX package in AAS metamodel `version`.

    Built in 3.0 with BaSyx 2.0.x (what IDTA 02012-1-0 was published for). 3.1 is the same
    content in the 3.1 schema: for the elements this Submodel uses the two differ only in
    the namespace, and the result is verified against 3.1 by aas-core3.1 (check_aasx)."""
    if version not in VERSIONS:
        raise ValueError(f"AAS version {version!r}: choose one of {VERSIONS}")
    store, files, aas_id = build_package(models, plant, **kw)
    buf = io.BytesIO()
    with aasx.AASXWriter(buf) as writer:
        writer.write_aas(aas_id, store, files, write_json=False)
    data = buf.getvalue()
    return data if version == "3.0" else _to_version(data, version)


def _to_version(data: bytes, version: str) -> bytes:
    """Rewrite the package's AAS environment (aasx/data.xml) into another metamodel version."""
    import zipfile

    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            content = src.read(info.filename)
            if info.filename == "aasx/data.xml":
                content = content.replace(_NS["3.0"].encode(), _NS[version].encode())
            dst.writestr(info, content)
    return out.getvalue()


def aasx_version(data: bytes) -> str | None:
    import zipfile

    xml = zipfile.ZipFile(io.BytesIO(data)).read("aasx/data.xml")[:400].decode("utf-8", "replace")
    return next((v for v, ns in _NS.items() if ns in xml), None)


def _verify_environment(data: bytes, version: str) -> list[str]:
    """aas-core (the AAS core team's reference implementation) verification of the AAS
    environment inside the package: schema + all metamodel constraints."""
    import zipfile

    xml = zipfile.ZipFile(io.BytesIO(data)).read("aasx/data.xml").decode("utf-8")
    if version == "3.0":
        from aas_core3 import verification, xmlization
    else:
        from aas_core3_1 import verification, xmlization
    try:
        env = xmlization.environment_from_str(xml)
    except Exception as exc:
        return [f"does not read as an AAS {version} environment: {exc}"]
    return [f"{e.path}: {e.cause}" for e in verification.verify(env)][:60]


def _verify_package(data: bytes) -> list[str]:
    """Package structure: every relationship target inside the zip exists."""
    import zipfile

    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    problems = []
    for rel in (n for n in names if n.endswith(".rels")):
        for target in re.findall(r'Target="/?([^"]+)"', z.read(rel).decode("utf-8")):
            if target not in names:
                problems.append(f"{rel}: missing part {target}")
    return problems


def check_aasx(data: bytes) -> tuple[bool, list[str]]:
    """Check an AASX package: (ok, problems). 3.0: the IDTA's aas-test-engines (package +
    metamodel) and aas-core3.0; 3.1 (which the IDTA test engine does not read yet): the
    package structure and aas-core3.1."""
    import zipfile

    version = aasx_version(data)
    if version is None:
        return False, ["no AAS 3.0 or 3.1 environment found in the package"]
    problems = _verify_package(data) + _verify_environment(data, version)
    if version == "3.0":
        from aas_test_engines.file import check_aasx_data

        result = check_aasx_data(zipfile.ZipFile(io.BytesIO(data)))
        if not result.ok():
            problems += [line for line in result.to_lines() if line.strip()][:60]
    return not problems, problems


def drawing_svg(diagram, result, source=None, xml: bytes | None = None) -> bytes | None:
    """The P&ID re-drawn from the DEXPI model (ModelRepresentation), as SVG bytes: the DEXPI
    file's own drawing (`xml` when made already; render/dexpi_svg.py), else generic symbols."""
    import tempfile

    from pid_dexpi_aas.render.build_diagram import build_diagram
    from pid_dexpi_aas.render.dexpi_svg import dexpi_svg, drawing_of
    from pid_dexpi_aas.render.render_svg import render_diagram_svg

    try:
        svg = dexpi_svg(xml or drawing_of(diagram, result, source))
        if svg:
            return svg.encode()
    except Exception:
        pass
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.svg"
            render_diagram_svg(build_diagram(diagram, result, result.model), path)
            return path.read_bytes()
    except Exception:  # the representation is optional (card. 0..1): never fail the package for it
        return None


def model_from_drawio(path: str | Path) -> DexpiModelInput:
    """A draw.io file through the same pipeline as the review app (draw.io mode, no edits):
    parse -> review rows -> DEXPI (verified by re-loading) -> SVG."""
    import tempfile

    from pid_dexpi_aas.drawio.parser import parse_drawio_file
    from pid_dexpi_aas.review import from_drawio, to_dexpi
    from pid_dexpi_aas.dexpi.serialize import save_and_verify

    path = Path(path)
    diagram, result = to_dexpi(from_drawio(parse_drawio_file(path)))
    with tempfile.TemporaryDirectory() as tmp:
        save_and_verify(result.model, Path(tmp), "model")
        xml = next(Path(tmp).glob("model*")).read_bytes()
    return DexpiModelInput(path.stem.strip(), xml, drawing_svg(diagram, result))


def main() -> None:
    """Bundle one plant's drawings into one AAS: python -m pid_dexpi_aas.aas.export ..."""
    import argparse

    ap = argparse.ArgumentParser(description="P&ID drawings (.drawio) -> AASX with the IDTA 02012-1-0 DEXPI Submodel")
    ap.add_argument("drawings", nargs="+", help=".drawio files of ONE plant (segment): one Model{00} each")
    ap.add_argument("-o", "--out", required=True, help="output .aasx")
    for name in PLANT_FIELDS:
        ap.add_argument(f"--{re.sub(r'(?<!^)(?=[A-Z])', '-', name).lower()}", dest=name, default=None)
    ap.add_argument("--base-iri", default=None, help="prefix of the AAS / Submodel / asset ids")
    ap.add_argument("--aas-version", choices=VERSIONS, default="3.0", help="AAS metamodel version (default 3.0)")
    args = ap.parse_args()
    plant = {k: getattr(args, k) for k in PLANT_FIELDS if getattr(args, k)}
    models = [model_from_drawio(d) for d in args.drawings]
    data = aasx_bytes(models, plant, version=args.aas_version, base_iri=args.base_iri)
    Path(args.out).write_bytes(data)
    ok, problems = check_aasx(data)
    n = sum(len(mapping_entries(m.dexpi_xml)) for m in models)
    print(f"{args.out}: AAS {args.aas_version}, {len(models)} DEXPI model(s), {n} mapped elements, "
          f"check {'passed' if ok else 'FAILED'}")
    for line in problems:
        print("  ", line)


if __name__ == "__main__":
    main()
