#!/usr/bin/env python3
"""
Version bump script for cartesia-line.

Bumps the version in pyproject.toml and updates all example pyproject.toml files
to pin the new version.

Usage:
    python scripts/bump_version.py --patch   # 0.1.9 -> 0.1.10
    python scripts/bump_version.py --minor   # 0.1.9 -> 0.2.0
    python scripts/bump_version.py --major   # 0.1.9 -> 1.0.0
    python scripts/bump_version.py --release # 0.2.4a2 -> 0.2.4
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys
from typing import Union


def get_current_version(pyproject_path: Path) -> Union[str, None]:
    """Extract version from pyproject.toml content."""
    content = pyproject_path.read_text()
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("version = "):
            start = line.find('"')
            end = line.rfind('"')
            if start != -1 and end != -1 and start < end:
                return line[start + 1 : end]
    return None


def strip_prerelease(patch_str: str) -> int:
    """Strip pre-release suffixes (e.g. '4a2' -> 4, '1rc1' -> 1)."""
    match = re.match(r"(\d+)", patch_str)
    if not match:
        raise ValueError(f"Cannot parse patch version: {patch_str}")
    return int(match.group(1))


def bump_version(version: str, bump_type: str) -> str:
    """Bump version according to semver."""
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version}. Expected X.Y.Z")

    major, minor, patch = int(parts[0]), int(parts[1]), strip_prerelease(parts[2])

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif bump_type == "release":
        return f"{major}.{minor}.{patch}"
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")


def update_root_pyproject(pyproject_path: Path, old_version: str, new_version: str) -> None:
    """Update version in root pyproject.toml."""
    content = pyproject_path.read_text()
    new_content = content.replace(f'version = "{old_version}"', f'version = "{new_version}"', 1)

    if content == new_content:
        raise RuntimeError(f"Failed to update version in {pyproject_path}")

    pyproject_path.write_text(new_content)


def update_example_pyproject(pyproject_path: Path, new_version: str) -> bool:
    """Update cartesia-line version pin in an example pyproject.toml.

    Returns True if the file was updated, False if no cartesia-line dependency was found.
    """
    content = pyproject_path.read_text()

    # Match cartesia-line with any version specifier (==X.Y.Z, >=X.Y.Z, etc.) or no version
    pattern = r'"cartesia-line[^"]*"'
    replacement = f'"cartesia-line=={new_version}"'

    new_content, count = re.subn(pattern, replacement, content)

    if count == 0:
        return False

    pyproject_path.write_text(new_content)
    return True


def find_example_pyprojects(root: Path) -> list[Path]:
    """Find all pyproject.toml files in examples/ and example_integrations/."""
    example_dirs = [root / "examples", root / "example_integrations"]
    pyprojects = []

    for example_dir in example_dirs:
        if example_dir.exists():
            pyprojects.extend(example_dir.rglob("pyproject.toml"))

    return sorted(pyprojects)


def main():
    parser = argparse.ArgumentParser(description="Bump version in pyproject.toml and update all examples")
    bump_group = parser.add_mutually_exclusive_group(required=True)
    bump_group.add_argument("--major", action="store_true", help="Bump major version (X.0.0)")
    bump_group.add_argument("--minor", action="store_true", help="Bump minor version (0.X.0)")
    bump_group.add_argument("--patch", action="store_true", help="Bump patch version (0.0.X)")
    bump_group.add_argument("--release", action="store_true", help="Strip pre-release tag (0.2.4a2 -> 0.2.4)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without making changes"
    )

    args = parser.parse_args()

    # Determine bump type
    if args.major:
        bump_type = "major"
    elif args.minor:
        bump_type = "minor"
    elif args.release:
        bump_type = "release"
    else:
        bump_type = "patch"

    # Find root directory (where this script lives in scripts/)
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    root_pyproject = root_dir / "pyproject.toml"

    if not root_pyproject.exists():
        print(f"❌ Error: {root_pyproject} not found")
        sys.exit(1)

    # Get current version
    current_version = get_current_version(root_pyproject)
    if not current_version:
        print("❌ Error: Could not extract version from pyproject.toml")
        sys.exit(1)

    # Calculate new version
    try:
        new_version = bump_version(current_version, bump_type)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print(f"Bumping version: {current_version} → {new_version} ({bump_type})")

    if args.dry_run:
        print("\n[DRY RUN] Would update the following files:\n")

    # Update root pyproject.toml
    if args.dry_run:
        print(f"  📦 {root_pyproject.relative_to(root_dir)}")
    else:
        update_root_pyproject(root_pyproject, current_version, new_version)
        print(f"  ✅ Updated {root_pyproject.relative_to(root_dir)}")

    # Find and update example pyprojects
    example_pyprojects = find_example_pyprojects(root_dir)

    if not example_pyprojects:
        print("\n⚠️  No example pyproject.toml files found")
    else:
        print(f"\nUpdating {len(example_pyprojects)} example pyproject.toml files:\n")

        for pyproject in example_pyprojects:
            rel_path = pyproject.relative_to(root_dir)
            if args.dry_run:
                print(f"  📦 {rel_path}")
            else:
                if update_example_pyproject(pyproject, new_version):
                    print(f"  ✅ Updated {rel_path}")
                else:
                    print(f"  ⚠️  Skipped {rel_path} (no cartesia-line dependency)")

    # Show commits since last version bump to help write the PR description
    try:
        last_bump_commit = subprocess.run(
            [
                "git",
                "log",
                "HEAD",
                "--grep=Bump version",
                "--grep=Version bump",
                "--grep=Bump to",
                "--format=%H",
                "-1",
            ],
            capture_output=True,
            text=True,
            cwd=root_dir,
        ).stdout.strip()
        if last_bump_commit:
            commit_log = subprocess.run(
                ["git", "log", f"{last_bump_commit}..HEAD", "--pretty=format:* %s (%h)"],
                capture_output=True,
                text=True,
                cwd=root_dir,
            ).stdout.strip()
            if commit_log:
                print(f"\nChanges since last version bump:\n{commit_log}")
    except Exception:
        print("❌ Error: Could not get commits since last version bump",)
        pass

    if args.dry_run:
        print("\n[DRY RUN] No files were modified")
    else:
        print(f"\n✅ Version bumped to {new_version}")
        print("\nNext steps:")
        print("  1. Review the changes: git diff")
        print("  2. Commit the changes: git add -A && git commit -m 'Bump version to " + new_version + "'")
        print("  3. Push and create a release")
        print("  4. In the PR description, include the changes since last version bump:")


if __name__ == "__main__":
    main()
