from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework_integration.lib.commands import framework_run_cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan (print) a Framework run command (does not execute).")
    parser.add_argument("--framework_config", required=True, help="Path to Framework config (yaml/json).")
    parser.add_argument("--framework_output_root", default="runs", help="Framework output dir (relative to Framework cwd).")
    parser.add_argument("--write", default="", help="If set, write the command into this .sh file.")
    args = parser.parse_args()

    cmd = framework_run_cmd(framework_config=str(args.framework_config), framework_output_root=str(args.framework_output_root))
    line = cmd.shell_line()
    print("# Run from:", cmd.cwd)
    print(line)
    if args.write:
        p = Path(args.write).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        script = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + f"cd {cmd.cwd}\n" + line + "\n"
        p.write_text(script, encoding="utf-8")
        print(f"Wrote: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
