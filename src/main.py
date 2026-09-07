#!/usr/bin/env python3
"""
Company Knowledge Base Assistant - Main Entry Point
"""

import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("build-index")

    dashboard_parser = subparsers.add_parser("dashboard")
    dashboard_parser.add_argument(
        "--tasks", help="Comma-separated task_ids to scope the aggregate to",
    )
    dashboard_parser.add_argument(
        "--compare", nargs="+", metavar="TASK_ID",
        help="Exactly 2 task_ids to compare",
    )

    timeline_parser = subparsers.add_parser("timeline")
    timeline_parser.add_argument("task_id")

    return parser


def _run_dashboard(args):
    from telemetry.dashboard import (
        compute_aggregate, compute_comparison, render_dashboard, render_comparison,
    )
    from rich.console import Console
    console = Console()

    if args.compare:
        if len(args.compare) != 2:
            print("❌ --compare requires exactly 2 task_ids (use --tasks for more)")
            sys.exit(1)
        rows = compute_comparison(None, args.compare[0], args.compare[1])
        console.print(render_comparison(rows, args.compare[0], args.compare[1]))
        return

    task_ids = args.tasks.split(",") if args.tasks else None
    aggregate = compute_aggregate(None, task_ids=task_ids)
    console.print(render_dashboard(aggregate))


def _run_timeline(task_id):
    from telemetry.timeline import fetch_timeline, render_timeline
    from rich.console import Console
    events = fetch_timeline(None, task_id)
    Console().print(render_timeline(task_id, events))


def _run_interactive():
    from assistant import CompanyKBAssistant
    from telemetry import context
    from telemetry.turn_summary import compute_turn_summary, format_turn_summary
    assistant = CompanyKBAssistant()

    print("=" * 60)
    print("🤖 Company Knowledge Base Assistant")
    print("=" * 60)
    print(f"\n🆔 Task ID: {assistant.task_id}")
    print("   (pass this to `python main.py timeline <task_id>` afterward)")
    print("\nAsk questions about company policies, procedures, and documentation.")
    print("Type 'exit' or 'quit' to stop\n")

    try:
        while True:
            query = input("❓ Question: ").strip()

            if not query:
                continue

            if query.lower() in {"exit", "quit", "q"}:
                print("\n👋 Goodbye!")
                break

            print("\n" + "─" * 60)
            print("🤖 Answer:\n")

            try:
                result = assistant.query(query, verbose=True)
                print(result["answer"])

                if result["sources"]:
                    print("\n📚 Sources:")
                    for src in result["sources"]:
                        print(f"  • {src}")

                if result["mcp_used"]:
                    print(f"\n🔧 Used MCP tool: {result['mcp_tool']}")

                turn_number = context.current_turn_number.get()
                summary = compute_turn_summary(None, assistant.task_id, turn_number)
                print(f"\n📊 {format_turn_summary(summary)}")

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

            print("─" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        assistant.close()


def main():
    """Main entry point for the assistant."""
    args = build_parser().parse_args()

    if args.command == "build-index":
        from rag.build_index import build_index
        build_index()
        return

    if args.command == "dashboard":
        _run_dashboard(args)
        return

    if args.command == "timeline":
        _run_timeline(args.task_id)
        return

    _run_interactive()


if __name__ == "__main__":
    main()
