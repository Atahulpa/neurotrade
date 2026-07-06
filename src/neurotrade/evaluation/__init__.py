"""Métriques et rapports — cœur pur (zéro I/O, zéro torch)."""

from .metrics import compute_metrics
from .report import generate_report

__all__ = ["compute_metrics", "generate_report"]
