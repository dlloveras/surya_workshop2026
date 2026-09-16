"""Tests for the nglo/pooling derivation.

Guards the bug where ``model.nglo`` had to be hand-kept in sync with ``model.pooling`` or
the long-short attention reshape failed deep in vendored code, on the first forward pass
(not at construction). ``nglo`` is now derived from ``pooling`` inside
``HelioSpectformer1D`` rather than being a config field.

Everything runs on CPU with a tiny backbone, so the suite is fast.
"""

import dataclasses

import pytest

from conftest import make_batch, make_model
from downstream_apps.template.configs import load_flare_config
from workshop_infrastructure.configs import ModelConfig


def test_model_config_has_no_nglo_field():
    assert "nglo" not in {f.name for f in dataclasses.fields(ModelConfig)}


def test_shipped_config_loads_and_has_no_nglo():
    cfg = load_flare_config("downstream_apps/template/configs/config_script.yaml")
    assert not hasattr(cfg.model, "nglo")


@pytest.mark.parametrize("pooling", ["class_token", "transformer", "attention", "global_average"])
def test_every_pooling_runs_a_forward_pass_with_the_derived_nglo(pooling):
    """The original bug only surfaced in forward(), not at construction."""
    model = make_model(pooling=pooling)
    output = model(make_batch())
    assert output.shape == (2,)
