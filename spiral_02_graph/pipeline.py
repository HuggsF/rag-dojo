"""Pipeline — orchestrates the full Graph RAG flow.

Analogy for a C dev:
- This is main() for Spiral 02
- ingest_graph() = parse structs from raw data and build a linked graph
- ask_hybrid() = traverse the graph AND search vectors, merge results
- Like spiral_01's pipeline but with a second blade: graph traversal

Usage:
    python -m spiral_02_graph.pipeline ingest-graph [./data/fastapi_docs/]
    python -m spiral_02_graph.pipeline ask "your question"
    python -m spiral_02_graph.pipeline stats
    python -m spiral_02_graph.pipeline visualize [entity_name]
"""

from __future__ import annotations

import sys
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spiral_01_chunks.generator import Generator
from spiral_01_chunks.retriever import Retriever
from spiral_01_chunks.store import VectorStore
from spiral_02_graph.entity_extractor import EntityExtractor
from spiral_02_graph.graph_store import GraphStore
from spiral_02_graph.hybrid_retriever import HybridRetriever
from spiral_02_graph.reranker import Reranker
from spiral_02_graph.visualizer import GraphVisualizer

console = Console()


class GraphRAGPipeline:
    """Full Graph RAG pipeline: Chunks → Extract → Graph → Hybrid Retrieve → Generate."""

    def __init__(self):
        self._vector_store = VectorStore()
        self._graph_store: GraphStore | None = None
        self._extractor: EntityExtractor | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._reranker: Reranker | None = None
        self._generator: Generator | None = None
        self._visualizer: GraphVisualizer | None = None

    @property
    def graph_store(self) -> GraphStore:
        """Lazy-load graph store (requires Neo4j running)."""
        if self._graph_store is None:
            self._graph_store = GraphStore()
        return self._graph_store

    @property
    def extractor(self) -> EntityExtractor:
        """Lazy-load entity extractor (requires GROQ_API_KEY)."""
        if self._extractor is None:
            self._extractor = EntityExtractor()
        return self._extractor

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        """Lazy-load hybrid retriever."""
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                vector_store=self._vector_store,
                graph_store=self.graph_store,
            )
        return self._hybrid_retriever

    @property
    def reranker(self) -> Reranker:
        """Lazy-load reranker."""
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    @property
    def generator(self) -> Generator:
        """Lazy-load generator (requires GROQ_API_KEY)."""
        if self._generator is None:
            self._generator = Generator()
        return self._generator

    @property
    def visualizer(self) -> GraphVisualizer:
        """Lazy-load visualizer."""
        if self._visualizer is None:
            self._visualizer = GraphVisualizer(self.graph_store)
        return self._visualizer

    def ingest_graph(self, path: str = "./data/fastapi_docs/") -> dict[str, Any]:
        """Build knowledge graph from already-ingested chunks.

        Prerequisites: spiral_01 must have ingested docs into ChromaDB first.

        Steps:
        1. Read chunks from ChromaDB (spiral_01 store)
        2. Extract entities/relations via LLM (Groq, zero cost)
        3. Persist to Neo4j
        4. Show stats
        """
        console.print("\n[bold blue]Building knowledge graph from chunks...[/]")

        # Get all chunks from vector store
        store_stats = self._vector_store.stats()
        total_chunks = store_stats["total_chunks"]

        if total_chunks == 0:
            console.print(
                "[red]No chunks in vector store. Run spiral_01 ingest first:[/]\n"
                "  python -m spiral_01_chunks.pipeline ingest ./data/fastapi_docs/"
            )
            return {"error": "no chunks"}

        console.print(f"[dim]Found {total_chunks} chunks in ChromaDB[/]")

        # Retrieve all chunks (query with empty-ish text, high top_k)
        retriever = Retriever(self._vector_store)
        all_results = retriever.retrieve("documentation", top_k=total_chunks, threshold=0.0)

        console.print(f"[dim]Retrieved {len(all_results)} chunks for extraction[/]")

        # Extract entities and relations
        console.print("[bold blue]Extracting entities and relations via LLM...[/]")
        console.print("[dim]This may take a few minutes (rate-limited for Groq free tier)[/]")

        start = time.time()
        total_entities = 0
        total_relations = 0

        for i, result in enumerate(all_results):
            chunk_id = f"{result.source}::chunk_{result.chunk_index}"

            try:
                extraction = self.extractor.extract_from_chunk(result.text, chunk_id)
                self.graph_store.persist_extraction(extraction)
                total_entities += len(extraction.entities)
                total_relations += len(extraction.relations)

                if (i + 1) % 5 == 0 or i == len(all_results) - 1:
                    console.print(
                        f"  [dim]Processed {i + 1}/{len(all_results)} chunks "
                        f"({total_entities} entities, {total_relations} relations)[/]"
                    )
            except Exception as e:
                console.print(f"  [yellow]Warning: chunk {i} failed: {e}[/]")

            # Rate limiting for Groq free tier
            if i < len(all_results) - 1:
                time.sleep(1.0)

        extract_time = time.time() - start

        # Show stats
        graph_stats = self.graph_store.stats()

        console.print(
            Panel(
                f"Extraction time: {extract_time:.1f}s\n"
                f"Total nodes: {graph_stats['total_nodes']}\n"
                f"Total edges: {graph_stats['total_edges']}\n"
                f"Labels: {graph_stats['labels']}\n"
                f"Relationship types: {graph_stats['relationship_types']}",
                title="[bold green]Graph Ingestion Complete",
            )
        )

        return {
            "extract_time_s": round(extract_time, 1),
            "total_entities": total_entities,
            "total_relations": total_relations,
            **graph_stats,
        }

    def ask_hybrid(
        self,
        question: str,
        top_k: int = 5,
        show_sources: bool = True,
    ) -> dict[str, Any]:
        """Ask using hybrid retrieval (vector + graph).

        Steps:
        1. Hybrid retrieve (vector + graph)
        2. Rerank
        3. Format context (with graph path info)
        4. Generate answer via Groq
        5. Display with Rich
        """
        console.print(f"\n[bold blue]Question:[/] {question}")

        # Hybrid retrieve
        start = time.time()
        context = self.hybrid_retriever.retrieve_with_context(question, top_k=top_k)
        retrieval_time = time.time() - start

        results = context["results"]
        traversal = context["traversal_info"]

        console.print(
            f"[dim]Retrieved {context['total_results']} hybrid results in {retrieval_time:.3f}s[/]"
        )
        console.print(
            f"[dim]  Vector: {context['vector_count']} results "
            f"(avg similarity: {context['avg_vector_similarity']:.3f})[/]"
        )
        console.print(
            f"[dim]  Graph:  {context['graph_count']} results "
            f"(avg score: {context['avg_graph_score']:.3f})[/]"
        )
        console.print(f"[dim]  {traversal}[/]")

        if not results:
            console.print("[yellow]No relevant results found.[/]")
            return {"answer": "No relevant context found.", "sources": []}

        # Rerank
        ranked = self.reranker.rerank(question, results, top_k=top_k)

        # Format context for LLM
        hybrid_results = results  # use pre-ranked for context
        formatted_context = HybridRetriever.format_context(hybrid_results)

        # Generate answer
        start = time.time()
        gen_result = self.generator.generate(formatted_context, question)
        gen_time = time.time() - start

        # Display answer
        console.print(Panel(gen_result.answer, title="[bold green]Answer"))

        if show_sources:
            table = Table(title="Sources")
            table.add_column("Score", style="cyan", width=8)
            table.add_column("Source", style="green", width=10)
            table.add_column("Type", style="magenta", width=8)
            table.add_column("Preview", style="dim")

            for r in ranked:
                preview = r.text[:70].replace("\n", " ")
                table.add_row(
                    f"{r.final_score:.3f}",
                    r.source,
                    "hybrid" if r.source == "hybrid" else r.source,
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
            "traversal_info": traversal,
            "sources": [
                {
                    "text": r.text[:100],
                    "source": r.source,
                    "score": r.final_score,
                }
                for r in ranked
            ],
        }

    def graph_stats(self) -> dict[str, Any]:
        """Show graph statistics."""
        stats = self.graph_store.stats()

        # Labels table
        label_table = Table(title="Node Labels")
        label_table.add_column("Label", style="cyan")
        label_table.add_column("Count", justify="right", style="green")
        for label, count in stats["labels"].items():
            label_table.add_row(label, str(count))

        # Relationship types table
        rel_table = Table(title="Relationship Types")
        rel_table.add_column("Type", style="cyan")
        rel_table.add_column("Count", justify="right", style="green")
        for rel_type, count in stats["relationship_types"].items():
            rel_table.add_row(rel_type, str(count))

        console.print(
            Panel(
                f"Total nodes: {stats['total_nodes']}\n"
                f"Total edges: {stats['total_edges']}\n"
                f"Dead ends: {stats['dead_ends']}",
                title="[bold blue]Graph Stats",
            )
        )
        console.print(label_table)
        console.print(rel_table)

        return stats

    def visualize(self, entity: str | None = None) -> str:
        """Export graph visualization to HTML."""
        if entity:
            path = self.visualizer.export_subgraph(entity)
            console.print(f"[green]Subgraph exported to: {path}[/]")
        else:
            path = self.visualizer.export_html()
            console.print(f"[green]Full graph exported to: {path}[/]")
        return path


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        console.print("[bold]Usage:[/]")
        console.print("  python -m spiral_02_graph.pipeline ingest-graph [path]")
        console.print("  python -m spiral_02_graph.pipeline ask <question>")
        console.print("  python -m spiral_02_graph.pipeline stats")
        console.print("  python -m spiral_02_graph.pipeline visualize [entity]")
        sys.exit(1)

    pipeline = GraphRAGPipeline()
    command = sys.argv[1]

    if command == "ingest-graph":
        path = sys.argv[2] if len(sys.argv) > 2 else "./data/fastapi_docs/"
        pipeline.ingest_graph(path)

    elif command == "ask":
        if len(sys.argv) < 3:
            console.print("[red]Error: provide a question[/]")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        pipeline.ask_hybrid(question)

    elif command == "stats":
        pipeline.graph_stats()

    elif command == "visualize":
        entity = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        pipeline.visualize(entity)

    else:
        console.print(f"[red]Unknown command: {command}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
