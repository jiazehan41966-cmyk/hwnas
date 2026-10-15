"""Read-only semantic audit of the authorized original-workbook note update.

WPS normalizes OOXML when saving an already-open workbook. This check preserves
all pre-existing values/formulas, rendered cell styles and link destinations,
while explicitly recording blank-font and serialization-only normalizations.
"""
from __future__ import annotations

from copy import copy
import hashlib
import json
import ntpath
from pathlib import Path
import sys
from urllib.parse import unquote

OUT = Path(__file__).resolve().parents[2] / "results/hardware_measurement_pilot_20261012"


def blank(value):
    return value in (None, "")


def color(value):
    if value is None:
        return None
    item = copy(value)
    if item.type == "rgb":
        item.rgb = "FF" + item.rgb[-6:]
    return item


def style(cell, attribute):
    value = copy(getattr(cell, attribute))
    if attribute == "font":
        if blank(cell.value):
            return None  # no text is rendered; all such changes are counted
        value.charset = None
        value.color = color(value.color)
    elif attribute == "fill":
        value.fgColor = color(value.fgColor)
        if value.patternType == "solid":
            from openpyxl.styles import Color
            value.bgColor = Color("FF000000")  # unused for solid fill
        else:
            value.bgColor = color(value.bgColor)
    elif attribute == "border":
        for edge in ("start", "end", "left", "right", "top", "bottom", "diagonal", "vertical", "horizontal"):
            side = getattr(value, edge)
            if side is None or side.style is None:
                setattr(value, edge, None)
            else:
                side.color = color(side.color)
    return value


def destination(link, original):
    if not link:
        return None
    target = unquote(link.target or "").replace("\\", "/")
    if target.startswith("file:///"):
        target = target[8:]
    if target and not "://" in target:
        if not ntpath.isabs(target):
            target = ntpath.join(str(Path(original).parent), target)
        target = ntpath.normcase(ntpath.normpath(target))
    return (target, link.location)


def audit():
    sys.path.insert(0, str(OUT / "_deps"))
    from openpyxl import load_workbook
    audit_path = OUT / "original_workbook_update.json"
    record = json.loads(audit_path.read_text(encoding="utf-8"))
    before, after = load_workbook(record["backup"]), load_workbook(record["original"])
    allowed = {(r["sheet"], r["cell"]): r["after"] for r in record["changes"]}
    values, styles, links, authorized_errors = [], [], [], []
    normalizations = {"blank_none_to_empty_string": 0, "blank_font_change": 0,
                      "equivalent_style_serialization_cells": 0, "equivalent_hyperlink_serialization": 0}
    for sheet in before:
        for row in sheet:
            for cell in row:
                key = (sheet.title, cell.coordinate)
                current = after[sheet.title][cell.coordinate]
                if key in allowed:
                    continue
                if cell.value != current.value:
                    if blank(cell.value) and blank(current.value):
                        normalizations["blank_none_to_empty_string"] += 1
                    else:
                        values.append(dict(sheet=key[0], cell=key[1], before=cell.value, after=current.value))
                serialization_only = False
                for attribute in ("font", "fill", "border", "alignment", "number_format", "protection"):
                    if copy(getattr(cell, attribute)) == copy(getattr(current, attribute)):
                        continue
                    if attribute == "font" and blank(cell.value):
                        normalizations["blank_font_change"] += 1
                    if style(cell, attribute) != style(current, attribute):
                        styles.append(dict(sheet=key[0], cell=key[1], attribute=attribute))
                    else:
                        serialization_only = True
                normalizations["equivalent_style_serialization_cells"] += int(serialization_only)
                if destination(cell.hyperlink, record["original"]) != destination(current.hyperlink, record["original"]):
                    links.append(dict(sheet=key[0], cell=key[1], before=destination(cell.hyperlink, record["original"]),
                                      after=destination(current.hyperlink, record["original"])))
                elif cell.hyperlink != current.hyperlink:
                    normalizations["equivalent_hyperlink_serialization"] += 1
    for (sheet, address), expected in allowed.items():
        observed = after[sheet][address].value
        if observed != expected and not (blank(observed) and blank(expected)):
            authorized_errors.append(dict(sheet=sheet, cell=address, expected=expected, observed=observed))
    frozen = {f"{column}{row}": after["先确定事项"][f"{column}{row}"].value
              for row in (11, 12, 14) for column in ("F", "G", "H")}
    merges_same = all(set(map(str, s.merged_cells.ranges)) == set(map(str, after[s.title].merged_cells.ranges)) for s in before)
    formulas_before = {(s.title, c.coordinate): c.value for s in before for row in s for c in row if c.data_type == "f"}
    formulas_after = {(s.title, c.coordinate): c.value for s in after for row in s for c in row if c.data_type == "f"}
    backup_sha = hashlib.sha256(Path(record["backup"]).read_bytes()).hexdigest()
    current_sha = hashlib.sha256(Path(record["original"]).read_bytes()).hexdigest()
    passed = not (values or styles or links or authorized_errors) and all(blank(v) for v in frozen.values())
    passed = passed and before.sheetnames == after.sheetnames and merges_same and formulas_before == formulas_after
    passed = passed and backup_sha == record["before_sha256"] and current_sha == record["after_sha256"]
    record.update(status="PASS" if passed else "REVIEW_REQUIRED", unexpected_value_changes=values,
                  unexpected_style_changes=styles, unexpected_hyperlink_changes=links,
                  authorized_cell_errors=authorized_errors, D06_D07_D09_F_G_H=frozen,
                  formal_freeze_written=False, sheet_names_unchanged=before.sheetnames == after.sheetnames,
                  merged_ranges_unchanged=merges_same, formulas_unchanged=formulas_before == formulas_after,
                  backup_sha256_verified=backup_sha == record["before_sha256"], normalization_counts=normalizations,
                  audit_scope="Existing values/formulas, visible cell formatting, resolved link destinations, authorized cells and blank freeze columns",
                  normalization_note="WPS normalized blank cells, OOXML color alpha/solid-fill backgrounds/empty borders and font charset; blank cells may have a different default font. These are recorded, not claimed byte-identical.")
    audit_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {key: record[key] for key in ("status", "unexpected_value_changes", "unexpected_style_changes",
            "unexpected_hyperlink_changes", "authorized_cell_errors", "normalization_counts", "formulas_unchanged",
            "backup_sha256_verified", "formal_freeze_written")}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
