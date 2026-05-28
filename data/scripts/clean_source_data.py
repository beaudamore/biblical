#!/usr/bin/env python3
"""
Clean source-data files by removing:
  - Project Gutenberg headers/footers
  - Publisher info, title pages, subscriber notices
  - Editor's prefaces and contents/table-of-contents pages
  - Sacred-texts.com navigation headers
  - Page markers (e.g., "p. 3")
  - IMPRIMI POTEST / NIHIL OBSTAT / IMPRIMATUR blocks
  - Translator prefaces (not the author's own prefaces)
  - Source attribution / metadata sections in ChristianFOSS
  - README files that are just metadata

Originals are NEVER modified. Cleaned files are written to
  ../source-clean/  (sibling to source-raw/ inside data/)
preserving the original directory structure.
"""

import os
import re
import shutil
from pathlib import Path

# All paths cascade from PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # biblical/
DATA_DIR     = PROJECT_ROOT / "data"
BASE         = DATA_DIR / "source-raw"                         # data/source-raw/
OUTPUT       = DATA_DIR / "source-clean"                       # data/source-clean/


# ─── Utility ────────────────────────────────

def write_clean(src: Path, text: str):
    """Write cleaned text to OUTPUT, mirroring the source path relative to BASE."""
    rel = src.relative_to(BASE)
    dest = OUTPUT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")


def copy_clean(src: Path):
    """Copy an already-clean file to OUTPUT unchanged."""
    rel = src.relative_to(BASE)
    dest = OUTPUT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


# ─────────────────────────────────────────────
# 1. Gutenberg .txt files
# ─────────────────────────────────────────────

def strip_gutenberg(text: str) -> str:
    """Strip Gutenberg header (before START marker) and footer (after END marker)."""
    start_pat = re.compile(
        r"^\*{3}\s*START OF (?:THE |THIS )?PROJECT GUTENBERG.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = start_pat.search(text)
    if m:
        text = text[m.end():]

    # Primary footer: *** END OF ...
    end_pat = re.compile(
        r"^\*{3}\s*END OF (?:THE |THIS )?PROJECT GUTENBERG.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = end_pat.search(text)
    if m:
        text = text[: m.start()]

    # Secondary footer: "End of Project Gutenberg's ..." (without ***)
    end2_pat = re.compile(
        r"^End of Project Gutenberg.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = end2_pat.search(text)
    if m:
        text = text[: m.start()]

    return text.strip() + "\n"


def find_augustine_content_start(text: str, filename: str) -> str:
    """
    Find where Augustine's actual text begins and strip ALL front matter.
    Per-file anchors avoid false matches on TOC lines.
    """
    lines = text.split("\n")

    if "City of God, Volume II" in filename:
        # Vol II – must be checked BEFORE Vol I ("Volume I" ⊂ "Volume II")
        for i, line in enumerate(lines):
            if "BOOK FOURTEENTH" in line:
                return "\n".join(lines[i:])

    elif "City of God, Volume I" in filename:
        # Vol I – Augustine's own preface (ALL-CAPS heading)
        for i, line in enumerate(lines):
            if "PREFACE, EXPLAINING HIS DESIGN" in line:
                return "\n".join(lines[i:])
        # Fallback
        for i, line in enumerate(lines):
            if "  AUGUSTINE CENSURES THE PAGANS" in line:
                return "\n".join(lines[i:])
        # Vol II – first real book heading
        for i, line in enumerate(lines):
            if "BOOK FOURTEENTH" in line:
                return "\n".join(lines[i:])

    elif "Donatist" in filename:
        for i, line in enumerate(lines):
            if "THE SEVEN BOOKS OF AUGUSTINE" in line:
                return "\n".join(lines[i:])

    elif "pg3296" in filename or "confessions" in filename.lower():
        for i, line in enumerate(lines):
            if re.match(r"^\s*BOOK\s+I\s*$", line.strip()):
                return "\n".join(lines[i:])

    else:
        pats = [
            re.compile(r"^\s*BOOK\s+I\s*$", re.IGNORECASE),
            re.compile(r"^\s*CHAPTER\s+[IVX1]", re.IGNORECASE),
        ]
        for i, line in enumerate(lines):
            if any(p.search(line.strip()) for p in pats):
                return "\n".join(lines[i:])

    return text  # fallback: unchanged


def clean_augustine_gutenberg(text: str, filename: str) -> str:
    """Full cleaning pipeline for Augustine Gutenberg .txt files."""
    text = strip_gutenberg(text)
    text = find_augustine_content_start(text, filename)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def clean_soliloquies(text: str) -> str:
    """Soliloquies: no START marker, only END markers."""
    end_pat = re.compile(
        r"^\*{3}\s*END OF (?:THE |THIS )?PROJECT GUTENBERG.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = end_pat.search(text)
    if m:
        text = text[: m.start()]
    # Secondary footer
    end2_pat = re.compile(
        r"^End of Project Gutenberg.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = end2_pat.search(text)
    if m:
        text = text[: m.start()]
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


# ─────────────────────────────────────────────
# 2. Sacred-texts.com extracted texts (augconf, fbe)
# ─────────────────────────────────────────────

def strip_sacred_texts_header(text: str) -> str:
    """Remove sacred-texts.com navigation boilerplate from the top."""
    lines = text.split("\n")
    nav_patterns = [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"^\s*Sacred Texts\s*$",
            r"^\s*Christianity\s*$",
            r"^\s*Bible\s*$",
            r"^\s*Apocrypha\s*$",
            r"^\s*Index\s*$",
            r"^\s*Previous\s*$",
            r"^\s*Next\s*$",
            r"at sacred-texts\.com",
            r"^\s*,\s*tr\.\s*by",
            r"^\s*,\s*by\s+",
            r"^\s*\[\d{4}(-\d{2,4})?\]",
        ]
    ]
    idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if any(p.search(stripped) for p in nav_patterns):
            idx = i + 1
            continue
        break
    return "\n".join(lines[idx:])


def strip_sacred_texts_footer(text: str) -> str:
    """Remove 'Next: ...' footer lines at the end."""
    lines = text.rstrip().split("\n")
    while lines and re.match(r"^\s*Next:\s", lines[-1].strip()):
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n"


def strip_page_markers(text: str) -> str:
    """Remove standalone 'p. NNN' page-number markers."""
    return re.sub(r"^p\.\s*\d+\s*$", "", text, flags=re.MULTILINE)


def strip_fbe_publisher_header(text: str) -> str:
    """Remove publisher info from Forgotten Books of Eden files."""
    lines = text.split("\n")
    cleaned = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if "The Forgotten Books of Eden" in stripped and not stripped.startswith("#"):
            skip = True
            continue
        if skip:
            if not stripped or re.match(r"^,?\s*by\s+", stripped, re.IGNORECASE):
                continue
            if "sacred-texts.com" in stripped:
                continue
            if re.match(r"^\[\d{4}\]", stripped):
                continue
            if re.match(r"^p\.\s*\d+", stripped):
                continue
            skip = False
        cleaned.append(line)
    return "\n".join(cleaned)


def clean_augconf(text: str) -> str:
    """Clean an Augustine Confessions chapter (sacred-texts source).

    These files have:  # Heading, then nav lines, then title/translator,
    then "BOOK <roman>" which starts the real content.
    """
    lines = text.split("\n")
    # Find the "BOOK <numeral>" line and start there
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*BOOK\s+[IVXLC]+\s*$", line.strip()):
            start = i
            break
    text = "\n".join(lines[start:])

    text = strip_sacred_texts_footer(text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def clean_fbe(text: str) -> str:
    """Clean a Forgotten Books of Eden file.

    Nav boilerplate and publisher info repeat at every chapter boundary.
    Strip ALL occurrences throughout the file, not just the top.
    """
    lines = text.split("\n")
    nav_words = {
        "Sacred Texts", "Bible", "Apocrypha", "Christianity",
        "Index", "Previous", "Next",
    }
    cleaned = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # Skip nav lines
        if stripped in nav_words:
            i += 1
            continue
        # Skip "at sacred-texts.com" attribution lines
        if "at sacred-texts.com" in stripped:
            i += 1
            continue
        # Skip "The Forgotten Books of Eden" publisher lines + trailing author/date
        if "The Forgotten Books of Eden" in stripped and not stripped.startswith("#"):
            i += 1
            # Skip following author/date/sacred-texts lines
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                if (re.match(r"^,?\s*by\s+", s, re.IGNORECASE) or
                    "sacred-texts.com" in s or
                    re.match(r"^\[\d{4}\]", s)):
                    i += 1
                    continue
                break
            continue
        # Skip "Next: ..." footer lines
        if re.match(r"^Next:\s", stripped):
            i += 1
            continue
        # Skip standalone page markers
        if re.match(r"^p\.\s*\d+\s*$", stripped):
            i += 1
            continue
        cleaned.append(lines[i])
        i += 1
    text = "\n".join(cleaned)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


# ─────────────────────────────────────────────
# 3. Alphonsus de Liguori / liguori .md files
# ─────────────────────────────────────────────

def clean_liguori(text: str) -> str:
    """
    Remove publication metadata from Liguori markdown files:
      - IMPRIMI POTEST / NIHIL OBSTAT / IMPRIMATUR blocks
      - Translator info and preface
      - "Also titled:" / "From the Italian of..." lines
    Preserve the # title and ## author heading plus actual content.
    """
    lines = text.split("\n")

    title_line = None
    author_line = None
    content_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            re.match(r"^#\s+", stripped)
            and not re.match(r"^##", stripped)
            and title_line is None
        ):
            title_line = i
            continue
        if re.match(r"^##\s+St\.\s+Alphonsus", stripped) and author_line is None:
            author_line = i
            continue
        if re.match(r"^(##\s+)?\d+\\?\.\s+\S", stripped):
            content_start = i
            break

    if title_line is not None and content_start is not None:
        cleaned = [lines[title_line], ""]
        if author_line is not None:
            cleaned.append(lines[author_line])
            cleaned.append("")
        cleaned.extend(lines[content_start:])
        text = "\n".join(cleaned)

    elif title_line is not None and author_line is not None:
        cleaned = [lines[title_line], ""]
        cleaned.append(lines[author_line])
        cleaned.append("")

        metadata_pats = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"^Also titled",
                r"^\*\*Also titled",
                r"^From the Italian",
                r"^\*\*From the Italian",
                r"^Translated by",
                r"^\*\*Translated by",
                r"^IMPRIMI POTEST",
                r"^\*\*IMPRIMI POTEST",
                r"^NIHIL OBSTAT",
                r"^\*\*NIHIL OBSTAT",
                r"^IMPRIMATUR",
                r"^\*\*IMPRIMATUR",
                r"^---$",
                r"^Provincial",
                r"^Diocesan",
                r"^Archbishop",
                r"^John\s+J",
                r"^Richard\s+J",
                r"^Nov\.\s+\d",
                r"^Jan\.\s+\d",
                r"^Oct\.\s+\d",
                r"^C\.SS\.R",
                r"^S\.J",
                r"^\*\*Preface\*\*",
                r"^##\s*Preface",
            ]
        ]

        in_content = False
        for j in range(max(title_line, author_line or 0) + 1, len(lines)):
            stripped = lines[j].strip()
            if not in_content:
                if not stripped:
                    continue
                if any(p.search(stripped) for p in metadata_pats):
                    continue
                if len(stripped) < 40 and not stripped.startswith("#"):
                    if re.match(
                        r"^(of|St\.|Thomas|THOMAS|Feast|C\.SS|"
                        r"January|February|March|April|May|June|"
                        r"July|August|September|October|November|December)",
                        stripped,
                    ):
                        continue
                in_content = True
            cleaned.append(lines[j])

        text = "\n".join(cleaned)

    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


# ─────────────────────────────────────────────
# 4. ChristianFOSS .md files
# ─────────────────────────────────────────────

def clean_christianfoss(text: str) -> str:
    """
    Remove metadata sections (Historical Context, Significance,
    Biblical Foundations, Source, License, Contributing, footer quotes).
    """
    lines = text.split("\n")
    cleaned = []
    skip_section = False
    meta_headings = [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"^##\s+Historical Context",
            r"^##\s+Significance",
            r"^##\s+Biblical Foundations",
            r"^##\s+Sources?\b",
            r"^##\s+License",
            r"^##\s+Contributing",
            r'^\*".*".*\*$',
        ]
    ]

    for line in lines:
        stripped = line.strip()
        if any(p.search(stripped) for p in meta_headings):
            skip_section = True
            continue
        if skip_section and re.match(r"^##\s+", stripped):
            if not any(p.search(stripped) for p in meta_headings):
                skip_section = False
        if skip_section:
            if stripped.startswith("---"):
                continue
            continue
        cleaned.append(line)

    # Remove italic subtitle near top
    final = []
    for i, line in enumerate(cleaned):
        stripped = line.strip()
        if i < 5 and re.match(r"^\*[^*]+\*$", stripped) and not stripped.startswith("**"):
            continue
        final.append(line)

    text = "\n".join(final)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Source Data Cleaner")
    print("=" * 60)
    print(f"  Reading from : {BASE}")
    print(f"  Writing to   : {OUTPUT}")
    print()

    # Wipe previous output so it's always a fresh clean
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    # ── Augustine Gutenberg texts ──
    aug_dir = BASE / "Bishop of Hippo Saint Augustine"
    if aug_dir.exists():
        print("── Bishop of Hippo Saint Augustine (Gutenberg) ──")
        for f in sorted(aug_dir.glob("*.txt")):
            raw = f.read_text(encoding="utf-8", errors="replace")
            if "Soliloquies" in f.name:
                cleaned = clean_soliloquies(raw)
            else:
                cleaned = clean_augustine_gutenberg(raw, f.name)
            write_clean(f, cleaned)
            print(f"  {f.name}  ({len(raw):,} → {len(cleaned):,})")
        print()

    # ── Extracted texts: augconf ──
    augconf_dir = BASE / "extracted_texts" / "augconf"
    if augconf_dir.exists():
        print("── extracted_texts/augconf ──")
        for f in sorted(augconf_dir.glob("*.txt")):
            raw = f.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_augconf(raw)
            write_clean(f, cleaned)
            print(f"  {f.name}  ({len(raw):,} → {len(cleaned):,})")
        print()

    # ── Extracted texts: fbe ──
    fbe_dir = BASE / "extracted_texts" / "fbe"
    if fbe_dir.exists():
        print("── extracted_texts/fbe ──")
        for f in sorted(fbe_dir.glob("*.txt")):
            raw = f.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_fbe(raw)
            write_clean(f, cleaned)
            print(f"  {f.name}  ({len(raw):,} → {len(cleaned):,})")
        print()

    # ── Extracted texts: kjv, asv, apo → already clean, copy ──
    for subdir in ("kjv", "asv", "apo"):
        d = BASE / "extracted_texts" / subdir
        if d.exists():
            count = 0
            for f in sorted(d.glob("*.txt")):
                copy_clean(f)
                count += 1
            print(f"── extracted_texts/{subdir}: {count} files copied (already clean) ──")
    print()

    # ── bib/ → already-clean markdown, copy recursively ──
    bib_dir = BASE / "bib"
    if bib_dir.exists():
        print("── bib/ (already-clean Bible texts) ──")
        for sub in sorted(bib_dir.iterdir()):
            if sub.is_dir():
                count = 0
                for f in sorted(sub.glob("*.md")):
                    copy_clean(f)
                    count += 1
                print(f"  bib/{sub.name}: {count} files copied")
        print()

    # ── Alphonsus de Liguori ──
    lig_dir = BASE / "Alphonsus de Liguori"
    if lig_dir.exists():
        print("── Alphonsus de Liguori ──")
        for f in sorted(lig_dir.glob("*.md")):
            if "Commentaries and Facts" in f.name:
                copy_clean(f)
                print(f"  {f.name}  (copied as-is, commentary)")
                continue
            raw = f.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_liguori(raw)
            write_clean(f, cleaned)
            print(f"  {f.name}  ({len(raw):,} → {len(cleaned):,})")
        print()

    # ── liguori/ ──
    lig2_dir = BASE / "liguori"
    if lig2_dir.exists():
        print("── liguori/ ──")
        for f in sorted(lig2_dir.glob("*.md")):
            raw = f.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_liguori(raw)
            write_clean(f, cleaned)
            print(f"  {f.name}  ({len(raw):,} → {len(cleaned):,})")
        print()

    # ── ChristianFOSS ──
    cf_dir = BASE / "ChristianFOSS"
    if cf_dir.exists():
        print("── ChristianFOSS ──")
        for f in sorted(cf_dir.glob("*.md")):
            if f.name == "README.md":
                print(f"  {f.name}  (skipped, README)")
                continue
            raw = f.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_christianfoss(raw)
            write_clean(f, cleaned)
            print(f"  {f.name}  ({len(raw):,} → {len(cleaned):,})")
        print()

    print("=" * 60)
    print(f"Done!  Clean files → {OUTPUT}")
    print("Originals in source-raw/ are untouched.")
    print("=" * 60)


if __name__ == "__main__":
    main()
