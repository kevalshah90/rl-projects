"""Tests for the static half of kernel_env/data.py. Mac only: no torch, no download, no execution.

Run:  uv run pytest tests/test_data.py -v
"""

from kernel_env.data import (MAX_ELEMENTS, Row, license_reason, parse, size_reason,
                             source_key, static_reason, structure_key)

# A small, valid KernelBook-style row, reused by every test.
CODE = '''
import torch
import torch.nn as nn


class SumAggregator(nn.Module):
    """Sums over dim 1."""

    def __init__(self):
        super(SumAggregator, self).__init__()

    def forward(self, neighbor):
        return torch.sum(neighbor, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
'''


def make_row(code: str = CODE, licenses: tuple = ("MIT",), name: str = "SumAggregator") -> Row:
    return Row(uuid=0, module_name=name, entry_point=name, python_code=code, triton_code="",
               licenses=licenses, synthetic=False, repo_link="")


# ---------- the good row passes everything ----------

def test_clean_row_is_kept():
    assert static_reason(make_row()) is None


# ---------- F2: license ----------

def test_permissive_licenses_pass():
    assert license_reason(make_row(licenses=("MIT", "Apache-2.0"))) is None


def test_copyleft_license_is_dropped():
    assert license_reason(make_row(licenses=("MIT", "GPL-3.0"))).startswith("license:")


def test_missing_license_is_dropped():
    assert license_reason(make_row(licenses=())) == "license: none listed"


# ---------- F3 / F4: inputs and length ----------

def test_oversized_inputs_are_dropped():
    big = CODE.replace("[4, 4, 4, 4]", "[2048, 2048]")          # 4,194,304 elements
    assert 2048 * 2048 > MAX_ELEMENTS
    assert size_reason(make_row(big), parse(big)).startswith("inputs:")


def test_no_input_tensors_are_dropped():
    none = CODE.replace("[torch.rand([4, 4, 4, 4])]", "[]")
    assert size_reason(make_row(none), parse(none)) == "inputs: get_inputs creates no tensors"


def test_too_long_is_dropped():
    long_code = CODE + "\n" * 250
    assert size_reason(make_row(long_code), parse(long_code)).startswith("length:")


# ---------- F1: duplicates ignore comments, docstrings and formatting ----------

def test_comment_and_docstring_changes_are_duplicates():
    edited = CODE.replace('"""Sums over dim 1."""', "# a different comment")
    edited = edited.replace("dim=1)", "dim = 1 )")                # spacing only
    assert source_key(edited) == source_key(CODE)


def test_real_code_change_is_not_a_duplicate():
    assert source_key(CODE.replace("dim=1", "dim=2")) != source_key(CODE)


# ---------- F6: KernelBench overlap ignores the class name ----------

def test_renamed_class_matches_kernelbench_model():
    kernelbench_style = CODE.replace("SumAggregator", "Model")
    assert structure_key(CODE, "SumAggregator") == structure_key(kernelbench_style, "Model")
