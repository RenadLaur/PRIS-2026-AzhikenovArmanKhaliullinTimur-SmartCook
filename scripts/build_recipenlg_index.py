#!/usr/bin/env python3
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender import ensure_recipenlg_search_index  # noqa: E402


def main():
    status = ensure_recipenlg_search_index(force_rebuild="--force" in sys.argv[1:])
    print("RecipeNLG search index status:")
    for key in ["ready", "needs_rebuild", "row_count", "path", "last_error", "built_at"]:
        print(f"- {key}: {status.get(key)}")


if __name__ == "__main__":
    main()
