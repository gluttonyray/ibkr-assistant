"""
Rich terminal dashboard — refreshes every cfg.refresh_interval seconds.

Layout:
  ┌─ Header ──────────────────────────────────────────┐
  │  IBKR Signal Assistant  |  timestamp              │
  ├─ Summary Table ────────────────────────────────────┤
  │  Symbol | Price | Score | Signal | Buy | Hold | Sell│
  ├─ Indicator Detail (per symbol) ───────────────────┤
  │  Name | Signal | Value | Label                     │
  └───────────────────────────────────────────────────┘
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import cfg
from src.indicators.aggregator import AggregateResult
from src.trading.data_feed import DataFeed

logger = logging.getLogger(__name__)

_SIGNAL_STYLE = {
    "BUY": "[bold green]BUY ▲[/bold green]",
    "SELL": "[bold red]SELL ▼[/bold red]",
    "HOLD": "[yellow]HOLD ─[/yellow]",
}

_REGIME_STYLE = {
    "TRENDING_CALM":     "[bold green]↗ TREND[/bold green]",
    "TRENDING_VOLATILE": "[yellow]↗ NOISY[/yellow]",
    "RANGING":           "[cyan]↔ RANGE[/cyan]",
    "CHOPPY_VOLATILE":   "[bold red]✗ CHOPPY[/bold red]",
    "UNKNOWN":           "[dim]?[/dim]",
}

_IND_SIGNAL_STYLE = {
    1: "[green]+1[/green]",
    0: "[dim]0[/dim]",
    -1: "[red]-1[/red]",
}


def _sym_display(symbol: str) -> str:
    """Format symbol with STK/FUT type tag."""
    _, is_futures = DataFeed._parse_symbol(symbol)
    if is_futures:
        return f"[bold yellow]{symbol}[/bold yellow] [dim yellow]FUT[/dim yellow]"
    return f"[bold cyan]{symbol}[/bold cyan] [dim]STK[/dim]"


def _score_bar(score: float, width: int = 20) -> str:
    """Visual ASCII bar for score ∈ [-1, +1]."""
    half = width // 2
    filled = int(round(abs(score) * half))
    if score > 0:
        bar = " " * half + "█" * filled + " " * (half - filled)
    else:
        bar = " " * (half - filled) + "█" * filled + " " * half
    mid = width // 2
    return f"[dim]|[/dim]{bar[:mid]}[bold white]|[/bold white]{bar[mid:]}[dim]|[/dim]"


class _DashboardRenderable:
    """
    Thin wrapper that implements Rich's renderable protocol.
    Rich Live calls __rich_console__ on its timer, so the dashboard
    only re-renders at the configured refresh rate — not on every data update.
    """

    def __init__(self, dashboard: "Dashboard") -> None:
        self._dashboard = dashboard

    def __rich_console__(self, console, options):
        yield from self._dashboard._render().__rich_console__(console, options)


class Dashboard:
    """
    Manages a Rich Live display that auto-refreshes every N seconds.
    Call `update(symbol, result)` from the bar handler to push new data.
    """

    def __init__(self) -> None:
        self._results: Dict[str, AggregateResult] = {}
        self._console = Console()
        self._live: Optional[Live] = None
        self._refresh_interval = cfg.refresh_interval

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Live display (blocking — run in separate task)."""
        self._live = Live(
            _DashboardRenderable(self),
            console=self._console,
            refresh_per_second=1 / self._refresh_interval,
            screen=False,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()

    # ------------------------------------------------------------------
    # Data update
    # ------------------------------------------------------------------

    def update(self, symbol: str, result: AggregateResult) -> None:
        """Store new data; rendering happens on Rich Live's timer, not here."""
        self._results[symbol] = result

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), size=3),
            Layout(self._summary_table(), name="summary"),
            Layout(self._indicator_tables(), name="details"),
        )
        return layout

    def _header(self) -> Panel:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        mode = "[bold red]AUTO-TRADE ON[/bold red]" if cfg.auto_trade else "[dim]paper[/dim]"
        title = Text.assemble(
            ("IBKR Signal Assistant", "bold cyan"),
            "  |  ",
            (ts, "dim"),
            "  |  ",
            mode,
            "  |  symbols: ",
            (", ".join(cfg.symbols), "bold"),
        )
        return Panel(Align.center(title), style="on default")

    def _summary_table(self) -> Panel:
        tbl = Table(
            show_header=True,
            header_style="bold magenta",
            expand=True,
            border_style="bright_black",
            title="[bold]Summary[/bold]",
        )
        tbl.add_column("Symbol", width=14)
        tbl.add_column("Price", justify="right", width=10)
        tbl.add_column("Regime", justify="center", width=14)
        tbl.add_column("Score", justify="center", width=26)
        tbl.add_column("Signal", justify="center", width=12)
        tbl.add_column("✓", justify="center", width=3)
        tbl.add_column("Buy", justify="right", style="green", width=5)
        tbl.add_column("Hold", justify="right", style="yellow", width=5)
        tbl.add_column("Sell", justify="right", style="red", width=5)

        if True:
            for sym in cfg.symbols:
                res = self._results.get(sym)
                sym_label = _sym_display(sym)
                if res is None:
                    tbl.add_row(sym_label, "—", "—", "—", "waiting...", "—", "—", "—", "—")
                    continue
                score_bar = _score_bar(res.score)
                score_text = f"{score_bar}  {res.score:+.3f}"
                signal_display = _SIGNAL_STYLE.get(res.final_signal, res.final_signal)
                regime_display = _REGIME_STYLE.get(res.regime, res.regime)
                confirmed = "✓" if res.confirmed else "·"
                tbl.add_row(
                    sym_label,
                    f"${res.price:.2f}",
                    regime_display,
                    score_text,
                    signal_display,
                    confirmed,
                    str(res.buy_count),
                    str(res.hold_count),
                    str(res.sell_count),
                )

        return Panel(tbl, border_style="bright_black")

    def _indicator_tables(self) -> Columns:
        """One compact table per symbol."""
        panels = []
        for sym in cfg.symbols:
            res = self._results.get(sym)
            sym_label = _sym_display(sym)
            if res is None:
                panels.append(Panel(f"[dim]Waiting for {sym}...[/dim]", title=sym_label))
                continue
            tbl = Table(
                show_header=True,
                header_style="bold blue",
                expand=True,
                border_style="bright_black",
                padding=(0, 1),
            )
            tbl.add_column("Indicator", no_wrap=True)
            tbl.add_column("Sig", justify="center", width=4)
            tbl.add_column("Value", justify="right", width=12)
            tbl.add_column("Note", style="dim", overflow="fold")

            for ind in res.indicators:
                sig_str = _IND_SIGNAL_STYLE.get(ind.signal, str(ind.signal))
                val_str = f"{ind.value:.4g}" if ind.value == ind.value else "—"  # NaN check
                tbl.add_row(ind.name, sig_str, val_str, ind.label)

            panels.append(
                Panel(
                    tbl,
                    title=f"{sym_label}  {_SIGNAL_STYLE.get(res.final_signal, '')}  [dim]{res.score:+.3f}[/dim]",
                    border_style="bright_black",
                )
            )
        return Columns(panels, equal=True, expand=True)
