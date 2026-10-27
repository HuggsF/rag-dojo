"""Unified CLI — single entry point for the full 6-spiral RAG pipeline.

Usage:
    python cli.py ask "What is FastAPI?"
    python cli.py ask "How does middleware work?" --no-graph
    python cli.py benchmark
    python cli.py benchmark --max-questions 5
    python cli.py improve
    python cli.py improve --max-cycles 2
    python cli.py dashboard

All modes go through the full stack:
    L0 Cache → L1 Keyword → L2 RAG (spiral_02) → L3 LLM (spiral_01)
    + Cognitive monitoring (spiral_04)
    + Cost tracking (spiral_05)
    + Self-improvement (spiral_06)
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

console = Console()


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question through the full pipeline."""
    from pipeline_factory import build_pipeline

    console.print(f"\n[bold]Query:[/bold] {args.question}")
    console.print("[dim]Building pipeline...[/dim]")

    pipeline = build_pipeline(
        enable_graph=not args.no_graph,
        enable_cognitive=not args.no_cognitive,
        use_ollama=getattr(args, "ollama", False),
    )

    result = pipeline.query(args.question)

    # Display result
    console.print(f"\n[bold green]Answer[/bold green] (layer: {result.layer}):")
    console.print(result.answer)
    console.print(
        f"\n[dim]Cost: ${result.cost:.6f} | Model: {result.model} | Cached: {result.cached}[/dim]"
    )

    # Cognitive alerts
    if "cognitive_analysis" in result.metadata:
        analysis = result.metadata["cognitive_analysis"]
        if analysis.get("alerts"):
            console.print("\n[bold yellow]Cognitive Alerts:[/bold yellow]")
            for alert in analysis["alerts"]:
                console.print(f"  ⚠ {alert}")

    # Metadata
    if args.verbose:
        console.print("\n[dim]Metadata:[/dim]")
        for k, v in result.metadata.items():
            if k != "cognitive_analysis":
                console.print(f"  {k}: {v}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run benchmark suite and display scorecard."""
    from pipeline_factory import build_answer_fn_from_pipeline, build_pipeline
    from spiral_06_state_of_art.benchmark_suite import BenchmarkSuite
    from spiral_06_state_of_art.scorer import Scorer

    console.print("\n[bold]Running benchmark...[/bold]")

    pipeline = build_pipeline(
        enable_graph=not args.no_graph,
        enable_cognitive=False,  # no monitoring during benchmark
        use_ollama=getattr(args, "ollama", False),
    )
    answer_fn = build_answer_fn_from_pipeline(pipeline)

    suite = BenchmarkSuite()
    questions = suite.questions[: args.max_questions] if args.max_questions else suite.questions
    suite = BenchmarkSuite(questions=questions)

    scorer = Scorer()
    from spiral_06_state_of_art.verifier import Verifier

    verifier = Verifier(suite=suite, scorer=scorer)
    report, scorecard = verifier.run_benchmark(answer_fn)

    # Display scorecard
    table = Table(title="Benchmark Scorecard")
    table.add_column("Dimension", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Weighted", justify="right")

    for dim in scorecard.dimensions:
        table.add_row(
            dim.name,
            f"{dim.value:.1f}",
            f"{dim.weight:.0%}",
            f"{dim.weighted():.1f}",
        )

    table.add_section()
    table.add_row("[bold]TOTAL[/bold]", "", "", f"[bold]{scorecard.total:.1f}[/bold]")

    console.print(table)

    # Report summary
    console.print(
        f"\n[dim]Questions: {report.total_questions} | "
        f"Avg latency: {report.mean_latency_s:.2f}s | "
        f"Total cost: ${report.total_cost:.6f}[/dim]"
    )


def cmd_improve(args: argparse.Namespace) -> None:
    """Run the self-improvement loop (PROPOSE → SOLVE → VERIFY)."""
    from pipeline_factory import (
        build_answer_fn_from_pipeline,
        build_cycle_manager,
        build_pipeline,
    )

    console.print("\n[bold]Starting self-improvement loop...[/bold]")

    pipeline = build_pipeline(
        enable_graph=not args.no_graph,
        enable_cognitive=False,
        use_ollama=getattr(args, "ollama", False),
    )
    answer_fn = build_answer_fn_from_pipeline(pipeline)
    manager, verifier = build_cycle_manager(pipeline, max_cycles=args.max_cycles)

    # Initial benchmark
    console.print("[dim]Running initial benchmark...[/dim]")
    _, initial_scorecard = verifier.run_benchmark(answer_fn)
    console.print(f"[dim]Initial score: {initial_scorecard.total:.1f}[/dim]")

    # Human checkpoint (optional)
    checkpoint_fn = None
    if not args.auto:

        def checkpoint_fn(cycle_num, cycle_result):
            console.print(f"\n[yellow]Cycle {cycle_num} complete: {cycle_result.outcome}[/yellow]")
            response = console.input("[bold]Continue? (y/n): [/bold]")
            return response.lower().startswith("y")

    # Run improvement loop
    result = manager.run(initial_scorecard, answer_fn, checkpoint_fn=checkpoint_fn)

    # Display results
    console.print("\n[bold]Improvement complete[/bold]")
    console.print(f"  Cycles run: {result.cycles_run}")
    console.print(f"  Total improvement: {result.total_improvement:+.2f}")
    console.print(f"  Stopped: {result.stopped_reason}")

    if result.final_scorecard:
        console.print(f"  Final score: {result.final_scorecard.total:.1f}")

    # Cycle details
    for cycle in result.cycles:
        icon = {"improved": "✓", "reverted": "✗", "no_proposal": "—", "neutral": "~"}.get(
            cycle.outcome, "?"
        )
        console.print(f"  [{icon}] Cycle {cycle.cycle_number}: {cycle.outcome}")


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest documents into ChromaDB and optionally Neo4j."""
    import os

    from rich.progress import Progress

    from spiral_01_chunks.pipeline import RAGPipeline

    path = args.path
    if not os.path.isdir(path):
        console.print(f"[red]Error: directory not found: {path}[/red]")
        return

    # --- Step 1: Chunk + embed into ChromaDB (spiral_01) ---
    console.print("\n[bold]Step 1: Ingesting chunks into ChromaDB...[/bold]")
    rag = RAGPipeline()

    # If --limit, we chunk the directory then trim
    from spiral_01_chunks.chunker import chunk_directory

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Chunking documents...", total=None)
        chunks = chunk_directory(path, strategy="recursive")
        progress.update(task, completed=100, total=100)

    if args.limit and args.limit > 0:
        chunks = chunks[: args.limit]
        console.print(f"[dim]Limited to first {args.limit} chunks[/dim]")

    console.print(f"[green]Chunked {len(chunks)} blocks[/green]")

    # Store in ChromaDB
    store = rag._store
    added = store.add_documents(chunks)
    console.print(f"[green]Stored {added} chunks in ChromaDB[/green]")

    # --- Step 2: Graph ingestion (optional) ---
    if not args.no_graph:
        console.print("\n[bold]Step 2: Extracting entities into Neo4j...[/bold]")
        try:
            if args.ollama:
                from spiral_02_graph.entity_extractor import OllamaEntityExtractor

                extractor = OllamaEntityExtractor(model="mistral:7b")
            else:
                from spiral_02_graph.entity_extractor import EntityExtractor

                extractor = EntityExtractor()

            from spiral_02_graph.graph_store import GraphStore

            graph_store = GraphStore()

            total_entities = 0
            total_relations = 0
            for i, chunk in enumerate(chunks):
                chunk_id = f"{chunk.source}::chunk_{chunk.chunk_index}"
                try:
                    extraction = extractor.extract_from_chunk(chunk.text, chunk_id)
                    graph_store.persist_extraction(extraction)
                    total_entities += len(extraction.entities)
                    total_relations += len(extraction.relations)

                    if (i + 1) % 5 == 0 or i == len(chunks) - 1:
                        console.print(
                            f"  [dim]Processed {i + 1}/{len(chunks)} chunks "
                            f"({total_entities} entities, {total_relations} relations)[/dim]"
                        )
                except Exception as e:
                    console.print(f"  [yellow]Warning: chunk {i} failed: {e}[/yellow]")
            console.print(
                f"[green]Graph ingestion complete: "
                f"{total_entities} entities, {total_relations} relations[/green]"
            )
        except Exception as e:
            console.print(f"[yellow]Graph ingestion skipped: {e}[/yellow]")
    else:
        console.print("\n[dim]Step 2: Graph ingestion skipped (--no-graph)[/dim]")

    console.print("\n[bold green]Ingestion complete![/bold green]")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Show cost tracking dashboard with optional metrics."""
    import json
    import os

    from spiral_05_economics.budget_tracker import BudgetTracker
    from spiral_05_economics.dashboard import Dashboard

    # If --with-query, run a sample query first to generate fresh metrics
    if getattr(args, "with_query", False):
        console.print("[dim]Running sample query to generate fresh metrics...[/dim]")
        from pipeline_factory import build_pipeline

        pipeline = build_pipeline(
            enable_graph=False,
            enable_cognitive=False,
            use_ollama=getattr(args, "ollama", False),
        )
        result = pipeline.query("What is FastAPI?")
        console.print(f"[dim]Sample query done (layer: {result.layer})[/dim]\n")

        # Save metrics summary to JSON
        if hasattr(pipeline, "metrics") and hasattr(pipeline.metrics, "summary"):
            metrics_data = pipeline.metrics.summary()
            metrics_path = os.path.join(".", "data", "metrics_summary.json")
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics_data, f, indent=2)
            console.print(f"[dim]Metrics saved to {metrics_path}[/dim]\n")

    tracker = BudgetTracker()
    dashboard = Dashboard(tracker=tracker)

    # Budget section
    table = Table(title="Cost Dashboard")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    summary = dashboard.summary()
    table.add_row("Daily Cost", f"${summary['daily_cost']:.4f}")
    table.add_row("Daily Budget", f"${summary['daily_budget']:.2f}")
    table.add_row("Monthly Cost", f"${summary['monthly_cost']:.4f}")
    table.add_row("Monthly Budget", f"${summary['monthly_budget']:.2f}")
    table.add_row("Daily Requests", str(summary["daily_requests"]))
    table.add_row("Monthly Requests", str(summary["monthly_requests"]))
    table.add_row("Alert Level", str(summary["alert_level"]))
    console.print(table)

    # Savings
    savings = dashboard.estimate_savings()
    console.print(
        f"\n[dim]Savings vs Sonnet: ${savings['saved_vs_sonnet']:.4f} "
        f"(actual: ${savings['actual_cost']:.4f}, "
        f"hypothetical: ${savings['hypothetical_sonnet_cost']:.4f})[/dim]"
    )

    # Metrics section (from file)
    metrics_path = os.path.join(".", "data", "metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics_data = json.load(f)

        console.print()
        metrics_table = Table(title="Pipeline Metrics (InMemoryMetrics)")
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", justify="right")

        # Latencies
        for op, data in metrics_data.get("latencies", {}).items():
            metrics_table.add_row(f"Avg {op} latency", f"{data.get('avg', 0):.4f}s")
            metrics_table.add_row(f"P99 {op} latency", f"{data.get('p99', 0):.4f}s")

        # Cache hit rates
        for layer, data in metrics_data.get("cache", {}).items():
            metrics_table.add_row(
                f"Cache hit rate ({layer})",
                f"{data.get('hit_rate', 0):.1%}",
            )

        # Token usage
        for model, count in metrics_data.get("tokens", {}).items():
            metrics_table.add_row(f"Tokens ({model})", str(count))

        console.print(metrics_table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Dojo — Unified 6-spiral pipeline",
        prog="rag-dojo",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="The question to ask")
    ask_parser.add_argument("--no-graph", action="store_true", help="Disable graph retrieval")
    ask_parser.add_argument(
        "--ollama", action="store_true", help="Use Ollama (local) instead of Groq"
    )
    ask_parser.add_argument(
        "--no-cognitive", action="store_true", help="Disable cognitive monitoring"
    )
    ask_parser.add_argument("-v", "--verbose", action="store_true", help="Show metadata")

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_parser.add_argument("--no-graph", action="store_true", help="Disable graph retrieval")
    bench_parser.add_argument(
        "--max-questions", type=int, default=0, help="Limit questions (0 = all)"
    )
    bench_parser.add_argument(
        "--ollama", action="store_true", help="Use Ollama (local) instead of Groq"
    )

    # improve
    improve_parser = subparsers.add_parser("improve", help="Run self-improvement loop")
    improve_parser.add_argument("--no-graph", action="store_true", help="Disable graph retrieval")
    improve_parser.add_argument("--max-cycles", type=int, default=3, help="Maximum cycles")
    improve_parser.add_argument("--auto", action="store_true", help="Skip human checkpoints")
    improve_parser.add_argument(
        "--ollama", action="store_true", help="Use Ollama (local) instead of Groq"
    )

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into ChromaDB + Neo4j")
    ingest_parser.add_argument("path", help="Path to documents directory")
    ingest_parser.add_argument(
        "--no-graph", action="store_true", help="Skip Neo4j entity extraction"
    )
    ingest_parser.add_argument(
        "--ollama", action="store_true", help="Use Ollama (local) instead of Groq for extraction"
    )
    ingest_parser.add_argument(
        "--limit", type=int, default=0, help="Only process first N chunks (0 = all)"
    )

    # dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Show cost tracking dashboard")
    dash_parser.add_argument(
        "--with-query",
        action="store_true",
        help="Run a sample query first to generate fresh metrics",
    )
    dash_parser.add_argument(
        "--ollama", action="store_true", help="Use Ollama (local) instead of Groq"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "ask": cmd_ask,
        "benchmark": cmd_benchmark,
        "improve": cmd_improve,
        "ingest": cmd_ingest,
        "dashboard": cmd_dashboard,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
