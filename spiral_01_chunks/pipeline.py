"""Pipeline — orchestrates the full RAG flow: ingest + ask.

Analogy for a C dev:
- This is main() — it wires everything together
- ingest() = compile + link (process docs into searchable form)
- ask() = runtime execution (query → retrieve → generate → answer)

Usage:
    python -m spiral_01_chunks.pipeline ingest ./data/fastapi_docs/
    python -m spiral_01_chunks.pipeline ask "Como criar um middleware no FastAPI?"
    python -m spiral_01_chunks.pipeline stats
    python -m spiral_01_chunks.pipeline compare "your question" --strategies fixed,recursive,sentence
"""

from __future__ import annotations

import sys
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spiral_01_chunks.chunker import STRATEGIES, chunk_directory
from spiral_01_chunks.config import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K
from spiral_01_chunks.generator import Generator
from spiral_01_chunks.retriever import Retriever
from spiral_01_chunks.store import VectorStore

console = Console()


class RAGPipeline:
    """Full RAG pipeline: Document → Chunk → Embed → Store → Retrieve → Generate."""

    def __init__(self):
        self._store = VectorStore()
        self._retriever = Retriever(self._store)
        self._generator: Generator | None = None

    @property
    def generator(self) -> Generator:
        """Lazy-load generator (requires GROQ_API_KEY)."""
        if self._generator is None:
            self._generator = Generator()
        return self._generator

    def ingest(
        self,
        path: str,
        strategy: str = "recursive",
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> dict[str, Any]:
        """Ingest documents from a directory into the vector store.

        This is the "compile" step — turns raw docs into searchable chunks.
        """
        console.print(f"\n[bold blue]Ingesting documents from:[/] {path}")
        console.print(
            f"[dim]Strategy: {strategy} | Chunk size: {chunk_size} | Overlap: {overlap}[/]"
        )

        start = time.time()
        chunks = chunk_directory(path, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
        chunk_time = time.time() - start

        console.print(f"[green]Chunked {len(chunks)} blocks in {chunk_time:.2f}s[/]")

        start = time.time()
        added = self._store.add_documents(chunks)
        embed_time = time.time() - start

        console.print(f"[green]Embedded and stored {added} chunks in {embed_time:.2f}s[/]")

        stats = self._store.stats()
        stats["chunks_created"] = len(chunks)
        stats["chunk_time_s"] = round(chunk_time, 2)
        stats["embed_time_s"] = round(embed_time, 2)
        stats["strategy"] = strategy

        console.print(
            Panel(
                f"Total chunks in store: {stats['total_chunks']}\n"
                f"Embedding model: {stats['embedding_model']}\n"
                f"Strategy: {strategy}",
                title="[bold green]Ingestion Complete",
            )
        )

        return stats  # type: ignore[no-any-return]

    def ask(
        self,
        question: str,
        top_k: int = TOP_K,
        show_sources: bool = True,
    ) -> dict[str, Any]:
        """Ask a question against the ingested documents.

        This is the "runtime" step:
        1. Embed the question
        2. Retrieve top-k similar chunks
        3. Format context
        4. Generate answer via LLM
        """
        console.print(f"\n[bold blue]Question:[/] {question}")

        # Retrieve
        start = time.time()
        retrieval = self._retriever.retrieve_with_context(question, top_k=top_k)
        retrieval_time = time.time() - start

        results = retrieval["results"]
        console.print(
            f"[dim]Retrieved {len(results)} chunks in {retrieval_time:.3f}s "
            f"(avg similarity: {retrieval['avg_similarity']:.3f})[/]"
        )

        if not results:
            console.print("[yellow]No relevant chunks found. Try ingesting more documents.[/]")
            return {"answer": "No relevant context found.", "sources": [], "tokens": 0}

        # Deduplicate near-identical chunks before formatting
        if len(results) >= 2:
            before_dedup = len(results)
            results = Retriever.deduplicate(results)
            if len(results) < before_dedup:
                console.print(
                    f"[dim]Deduplicated: {before_dedup} → {len(results)} chunks "
                    f"(removed {before_dedup - len(results)} near-duplicates)[/]"
                )

        # Format context
        context = Retriever.format_context(results)

        # Generate
        start = time.time()
        gen_result = self.generator.generate(context, question)
        gen_time = time.time() - start

        # Display
        console.print(Panel(gen_result.answer, title="[bold green]Answer"))

        if show_sources:
            table = Table(title="Sources")
            table.add_column("Similarity", style="cyan", width=12)
            table.add_column("Source", style="green")
            table.add_column("Preview", style="dim")

            for r in results:
                preview = r.text[:80].replace("\n", " ")
                table.add_row(
                    f"{r.similarity:.3f}",
                    r.source,
                    preview + "...",
                )
            console.print(table)

        console.print(
            f"\n[dim]Model: {gen_result.model} | "
            f"Tokens: {gen_result.total_tokens} | "
            f"Cost: ${gen_result.estimated_cost:.4f} | "
            f"Retrieval: {retrieval_time:.3f}s | "
            f"Generation: {gen_time:.3f}s[/]"
        )

        return {
            "answer": gen_result.answer,
            "model": gen_result.model,
            "tokens": gen_result.total_tokens,
            "cost": gen_result.estimated_cost,
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": round(gen_time, 3),
            "sources": [
                {"source": r.source, "similarity": r.similarity, "preview": r.text[:100]}
                for r in results
            ],
        }

    def stats(self) -> dict[str, Any]:
        """Show current store statistics."""
        s = self._store.stats()
        console.print(
            Panel(
                f"Collection: {s['collection']}\n"
                f"Total chunks: {s['total_chunks']}\n"
                f"Embedding model: {s['embedding_model']}\n"
                f"Similarity metric: {s['similarity_metric']}",
                title="[bold blue]Store Stats",
            )
        )
        return s  # type: ignore[no-any-return]

    def compare_strategies(
        self,
        question: str,
        path: str = "./data/fastapi_docs/",
        strategies: list[str] | None = None,
    ) -> None:
        """Compare chunking strategies for the same question.

        Bonus challenge: which strategy returns better results?
        """
        strategies = strategies or list(STRATEGIES.keys())
        console.print(f"\n[bold blue]Comparing strategies:[/] {strategies}")
        console.print(f"[bold blue]Question:[/] {question}\n")

        table = Table(title="Strategy Comparison")
        table.add_column("Strategy", style="cyan")
        table.add_column("Chunks", justify="right")
        table.add_column("Avg Similarity", justify="right")
        table.add_column("Top Similarity", justify="right")
        table.add_column("Answer Preview", width=50)

        for strategy in strategies:
            # Create isolated store per strategy
            store = VectorStore(
                collection_name=f"compare_{strategy}",
            )
            store.clear()

            chunks = chunk_directory(path, strategy=strategy)
            store.add_documents(chunks)

            retriever = Retriever(store)
            results = retriever.retrieve(question)

            avg_sim = sum(r.similarity for r in results) / len(results) if results else 0
            top_sim = results[0].similarity if results else 0

            # Generate answer
            context = Retriever.format_context(results)
            try:
                gen = self.generator.generate(context, question)
                answer_preview = gen.answer[:100] + "..."
            except Exception:
                answer_preview = "(generation skipped — no API key)"

            table.add_row(
                strategy,
                str(len(chunks)),
                f"{avg_sim:.3f}",
                f"{top_sim:.3f}",
                answer_preview,
            )

            # Clean up
            store.clear()

        console.print(table)


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        console.print("[bold]Usage:[/]")
        console.print("  python -m spiral_01_chunks.pipeline ingest <path>")
        console.print("  python -m spiral_01_chunks.pipeline ask <question>")
        console.print("  python -m spiral_01_chunks.pipeline stats")
        console.print("  python -m spiral_01_chunks.pipeline compare <question>")
        sys.exit(1)

    pipeline = RAGPipeline()
    command = sys.argv[1]

    if command == "ingest":
        if len(sys.argv) < 3:
            console.print("[red]Error: provide path to documents directory[/]")
            sys.exit(1)
        pipeline.ingest(sys.argv[2])

    elif command == "ask":
        if len(sys.argv) < 3:
            console.print("[red]Error: provide a question[/]")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        pipeline.ask(question)

    elif command == "stats":
        pipeline.stats()

    elif command == "compare":
        if len(sys.argv) < 3:
            console.print("[red]Error: provide a question to compare[/]")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        pipeline.compare_strategies(question)

    else:
        console.print(f"[red]Unknown command: {command}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
