"""Command line interface: build an index, then query it."""

from __future__ import annotations

import argparse
import sys

from .index import Index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="localrag", description="Local BM25 document search")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="index a directory of .txt/.md/.rst files")
    build.add_argument("directory")
    build.add_argument("-o", "--output", default="index.json")

    query = sub.add_parser("query", help="search a saved index")
    query.add_argument("text")
    query.add_argument("-i", "--index", default="index.json")
    query.add_argument("-k", type=int, default=3, help="number of results")

    args = parser.parse_args(argv)

    if args.command == "build":
        index = Index()
        count = index.add_directory(args.directory)
        index.save(args.output)
        print(f"Indexed {count} chunks -> {args.output}")
        return 0

    index = Index.load(args.index)
    results = index.search(args.text, k=args.k)
    if not results:
        print("No matches.")
        return 1
    for rank, r in enumerate(results, 1):
        snippet = r.chunk.text[:160].replace("\n", " ")
        print(f"{rank}. [{r.score:.3f}] {r.chunk.source}#{r.chunk.position}\n   {snippet}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
