from app.evaluation.evaluator import EvaluationReport


class EvaluationReporter:
    """Format and render evaluation benchmark results as formatted Markdown / ASCII tables."""

    @staticmethod
    def format_report(report: EvaluationReport) -> str:
        """Return a formatted Markdown report table for display in terminal or documentation."""
        header_lines = [
            f"# SearchIQ Retrieval Benchmark Report",
            f"**Dataset Name:** {report.dataset_name}",
            f"**Total Test Cases:** {report.total_test_cases}",
            "",
            "| Strategy | Recall@5 | Recall@25 | MRR@5 | Precision@5 | Avg Time |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        table_rows = []
        for s in report.strategies:
            r5_str = f"{s.metrics.recall_at_5:.3f}"
            r25_str = (
                f"{s.metrics.recall_at_25:.3f}"
                if s.metrics.recall_at_25 is not None
                else "N/A"
            )
            mrr5_str = f"{s.metrics.mrr_at_5:.3f}"
            p5_str = f"{s.metrics.precision_at_5:.3f}"
            lat_str = f"{s.avg_latency_ms:.1f}ms"

            table_rows.append(
                f"| **{s.strategy_name}** | {r5_str} | {r25_str} | {mrr5_str} | {p5_str} | {lat_str} |"
            )

        return "\n".join(header_lines + table_rows)

    @staticmethod
    def format_console(report: EvaluationReport) -> str:
        """Return a formatted ASCII table suitable for stdout terminal display."""
        border = "=" * 88
        sep = "-" * 88

        title = f"SEARCHIQ RETRIEVAL BENCHMARK REPORT ({report.dataset_name})"
        meta = f"Total Test Cases Evaluated: {report.total_test_cases}"

        header = f"{'Strategy':<26} {'Recall@5':<11} {'Recall@25':<11} {'MRR@5':<10} {'Precision@5':<13} {'Avg Latency':<11}"

        rows = []
        for s in report.strategies:
            r5_str = f"{s.metrics.recall_at_5:.3f}"
            r25_str = (
                f"{s.metrics.recall_at_25:.3f}"
                if s.metrics.recall_at_25 is not None
                else "N/A"
            )
            mrr5_str = f"{s.metrics.mrr_at_5:.3f}"
            p5_str = f"{s.metrics.precision_at_5:.3f}"
            lat_str = f"{s.avg_latency_ms:.1f}ms"

            rows.append(
                f"{s.strategy_name:<26} {r5_str:<11} {r25_str:<11} {mrr5_str:<10} {p5_str:<13} {lat_str:<11}"
            )

        return "\n".join(
            [
                border,
                f"{title:^88}",
                f"{meta:^88}",
                border,
                header,
                sep,
                *rows,
                border,
            ]
        )
