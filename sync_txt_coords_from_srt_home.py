import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path


HOME_RE = re.compile(
    r"HOME\((?P<lon_dir>[EW]):\s*(?P<lon>\d+(?:\.\d+)?),\s*(?P<lat_dir>[NS]):\s*(?P<lat>\d+(?:\.\d+)?)\)"
)
LAT_RE = re.compile(r"^(?P<prefix>\s*lat\s*:\s*)(?P<value>.*?)(?P<suffix>\s*)$", re.IGNORECASE | re.MULTILINE)
LON_RE = re.compile(r"^(?P<prefix>\s*lon\s*:\s*)(?P<value>.*?)(?P<suffix>\s*)$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class UpdateResult:
    txt_path: Path
    srt_path: Path
    changed: bool
    reason: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text_preserve_newline(path: Path, content: str, original: str) -> None:
    # Preserve newline style if possible
    newline = "\r\n" if "\r\n" in original else "\n"
    normalized = content.replace("\r\n", "\n").replace("\n", newline)
    path.write_text(normalized, encoding="utf-8")


def _apply_dir(value: str, direction: str) -> str:
    direction = direction.upper()
    if direction in ("S", "W"):
        return f"-{value.lstrip('-')}"
    return value.lstrip('+')


def extract_home_lat_lon(srt_text: str) -> tuple[str, str] | None:
    m = HOME_RE.search(srt_text)
    if not m:
        return None
    lat = _apply_dir(m.group("lat"), m.group("lat_dir"))
    lon = _apply_dir(m.group("lon"), m.group("lon_dir"))
    return (lat, lon)


def update_txt_lat_lon(txt_text: str, lat: str, lon: str) -> tuple[str, bool, str]:
    changed = False
    reason_parts: list[str] = []

    def repl_lat(m: re.Match[str]) -> str:
        nonlocal changed
        current = (m.group("value") or "").strip()
        if current != lat:
            changed = True
            reason_parts.append(f"lat {current!r}->{lat!r}")
        return f"{m.group('prefix')}{lat}{m.group('suffix')}"

    def repl_lon(m: re.Match[str]) -> str:
        nonlocal changed
        current = (m.group("value") or "").strip()
        if current != lon:
            changed = True
            reason_parts.append(f"lon {current!r}->{lon!r}")
        return f"{m.group('prefix')}{lon}{m.group('suffix')}"

    new_text = LAT_RE.sub(repl_lat, txt_text, count=1)
    if new_text == txt_text:
        # No lat field; append.
        changed = True
        reason_parts.append("lat appended")
        new_text = new_text.rstrip("\r\n") + f"\nlat: {lat}\n"

    newer_text = LON_RE.sub(repl_lon, new_text, count=1)
    if newer_text == new_text:
        changed = True
        reason_parts.append("lon appended")
        newer_text = newer_text.rstrip("\r\n") + f"lon: {lon}\n"

    reason = ", ".join(reason_parts) if reason_parts else "no change"
    return newer_text, changed, reason


def process_pair(txt_path: Path, dry_run: bool) -> UpdateResult:
    srt_path = txt_path.with_suffix(".srt")
    if not srt_path.exists():
        return UpdateResult(txt_path, srt_path, False, "missing .srt")

    srt_text = _read_text(srt_path)
    home = extract_home_lat_lon(srt_text)
    if home is None:
        return UpdateResult(txt_path, srt_path, False, "no HOME() in .srt")

    lat, lon = home
    original_txt = _read_text(txt_path)
    updated_txt, changed, reason = update_txt_lat_lon(original_txt, lat=lat, lon=lon)

    if changed and not dry_run:
        _write_text_preserve_newline(txt_path, updated_txt, original_txt)

    return UpdateResult(txt_path, srt_path, changed, reason)


def iter_txt_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync lat/lon in *.txt files from paired *.srt HOME(W,S) coordinates.")
    parser.add_argument(
        "--root",
        default=str(Path("inputs") / "raw"),
        help="Root folder to scan for .txt files (default: inputs/raw)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    txt_files = iter_txt_files(root)
    if not txt_files:
        print(f"No .txt files found under {root}")
        return 0

    changed_count = 0
    missing_srt = 0
    missing_home = 0

    for txt_path in txt_files:
        res = process_pair(txt_path, dry_run=args.dry_run)
        if res.reason == "missing .srt":
            missing_srt += 1
            continue
        if res.reason == "no HOME() in .srt":
            missing_home += 1
            continue
        if res.changed:
            changed_count += 1
            print(f"CHANGED {res.txt_path} ({res.reason})")

    print(
        "Summary:",
        f"txt_files={len(txt_files)}",
        f"changed={changed_count}",
        f"missing_srt={missing_srt}",
        f"missing_home={missing_home}",
        f"dry_run={args.dry_run}",
        sep=" ",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
