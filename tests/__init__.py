"""Test package marker.

This file makes ``tests`` a *regular* package. Do not delete it.

Without it, ``tests`` is a namespace package, and Python prefers any regular package
named ``tests`` found anywhere on ``sys.path`` (Google Colab ships one in
``dist-packages``), so ``from tests.fixtures... import ...`` would import the wrong
package. As a regular package at the project root (which ``conftest.py`` puts first on
``sys.path``), it wins. Regression test: ``tests/test_import_isolation.py``.
"""
