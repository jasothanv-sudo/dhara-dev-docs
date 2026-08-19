# DHARA — developer documentation archive

The technical/developer manual for DHARA's solver — theory, architecture,
and installation — as distinct from the **customer-facing** user manual
that ships inside the product itself (`dhara_jaso/manual/DHARA_User_Manual.md`,
rendered live in the desktop app's Help Center). This repo exists to keep
every version of this document in one place; it isn't itself part of the
shipped product.

## The core manual — three versions, oldest to newest

| File | Scope | Size |
|---|---|---|
| `DHARA_compressible_documentation.md` | Compressible module only — v1 | 15.6 KB |
| `DHARA_compressible_documentationv2.md` | Compressible module only — v2, expanded | 20.7 KB |
| `dhara_document.md` | **All four modules** (Compressible, Convection, Rayleigh–Bénard, Quantum) — installation guide, architecture, and per-module theory, chaptered | 44.7 KB |

`dhara_document.md` is the current, most complete version — it grew from
a single-module writeup into a full framework manual, chapter by chapter
(Chapter 0: installation, Chapter 1: architecture, Chapter 2 onward:
per-module theory). The two compressible-only files are kept as the
earlier stages of that same document's history, not duplicated content to
choose between.

## Two focused reference documents

| File | Scope |
|---|---|
| `DHARA_Differential_Equations.md` | The mathematical equations solved by the 2D solver, gathered in one place |
| `DHARA_Line_by_Line_Analysis.md` | A line-by-line explanation of the 2D solver's code, paired with the equations it implements |

## `dhara_project_handoff.txt` — this whole effort is unfinished

A handoff note for continuing this documentation project in a fresh
session. As of 2026-08-19, per its own "Suggested next steps" section,
**chapters 0 and 5–8 of the intended full structure are not yet drafted** —
open questions are flagged (the Convection module's exact formulation, a
Quantum linear-vs-nonlinear ambiguity, a Chapter 5/8 overlap decision, and
real `para.py` parameter names still needed for an accurate Chapter 6).
Read this file before assuming the manual above is complete.

## Provenance

Recovered 2026-08-19 from a Windows Downloads folder and two now-retired
repositories. `DHARA_compressible_documentation.md` and
`DHARA_compressible_documentationv2.md` came from `Dhara-J` (the original
pre-rename DHARA package repo, superseded by `DHARA_Local` and deleted
after these were extracted). `dhara_document.md` came from `public_one` (a
repo that also held an identical copy of `DHARA_compressible_documentationv2.md`,
named `document.md` — confirmed byte-for-byte identical before being left
out of this collection as a straight duplicate). The two reference
documents and the handoff file were sitting in `Downloads/`, alongside
several redundant format-converted/re-downloaded copies of
`DHARA_compressible_documentationv2.md` (`.docx`, `.pdf`, a browser
double-download) that were left out as pure duplicates of what's already
here.

See [`vayusoft-docs`](https://github.com/jasothanv-sudo/vayusoft-docs) for
the full map of every other DHARA/TARANG repository.
