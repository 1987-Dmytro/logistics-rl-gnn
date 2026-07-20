"""Smoke-тест: пакет импортируется и версия совпадает."""

import logistics_rl_gnn


def test_version():
    assert logistics_rl_gnn.__version__ == "0.0.0"
