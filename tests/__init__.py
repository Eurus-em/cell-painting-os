"""Test package for CellPainting-Claw.

This file is load-bearing, not decorative. Without it ``tests/`` is only a
namespace-package candidate: Python records it while scanning ``sys.path`` and
keeps looking, so any regular package named ``tests`` installed elsewhere wins.
``flow_matching`` (a CellFlux dependency) ships exactly such a package into
site-packages, which made every ``tests.test_*`` module unimportable and broke
``cellpaint-pipeline-tests`` entirely.

With this file present, ``tests`` is a regular package found at the repo root,
which sits earlier on ``sys.path`` than site-packages.
"""
