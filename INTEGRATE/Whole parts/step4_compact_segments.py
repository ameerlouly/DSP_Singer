"""Compatibility entry point for Step 4's former compact generator.

PDF Step 4 requires real pauses and original natural edge context. This entry
point now uses the same continuous-source planner as the UI; it never stitches
non-contiguous speech or adds synthetic silence. Old sessions remain readable
for inspection/conversion, but their audio must pass PDF export checks.
"""
from step4_cut_segments import export_candidates
