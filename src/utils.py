"""Provenance of every product: a .json next to it."""
import json
from datetime import datetime
from pathlib import Path


def _file_info(path):
    st = Path(path).stat()
    return {"path": str(Path(path).resolve()), "bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}


def write_provenance(output, script, inputs, params, counts):
    info = {
        "script": script,
        "date": datetime.now().isoformat(timespec="seconds"),
        "inputs": {k: _file_info(v) for k, v in inputs.items()},
        "params": params,
        "counts": counts,
    }
    Path(output).with_suffix(".json").write_text(json.dumps(info, indent=2))
