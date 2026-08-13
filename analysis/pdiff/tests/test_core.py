"""Unit tests for analysis.pdiff — identity, symmetry, bounds, sparsity.

Runs under plain unittest (no pytest dependency).

    python -m unittest analysis.pdiff.tests.test_core
"""

import json
import unittest
from pathlib import Path

from analysis.pdiff import (
    Diff,
    TrajectoryView,
    build_reference_vocabulary,
    diff,
    edit_distance,
    module_distance,
    ood_items,
    ood_score,
    signature,
    token_distance,
    view_from_patch,
    view_from_trace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESOLVED_LITE = PROJECT_ROOT / "output" / "resolved_traces_lite_full.jsonl"


def _sample_view(edits: list[str], tokens: list[str], scopes: list[str], modules: list[str]) -> TrajectoryView:
    return TrajectoryView(
        tokens=tokens,
        edits=frozenset(edits),
        scopes=frozenset(scopes),
        modules=frozenset(modules),
    )


class TestDistances(unittest.TestCase):
    def test_token_distance_identity(self):
        self.assertEqual(token_distance(["a", "b"], ["a", "b"]), 0.0)

    def test_token_distance_disjoint(self):
        self.assertEqual(token_distance(["a"], ["b"]), 1.0)

    def test_token_distance_symmetric(self):
        a = ["x", "y", "z"]
        b = ["x", "w"]
        self.assertAlmostEqual(token_distance(a, b), token_distance(b, a))

    def test_token_distance_empty_empty(self):
        self.assertEqual(token_distance([], []), 0.0)

    def test_token_distance_empty_one_side(self):
        self.assertEqual(token_distance([], ["a", "b"]), 1.0)
        self.assertEqual(token_distance(["a", "b"], []), 1.0)

    def test_token_distance_bounds(self):
        for a, b in [(["a"], ["b", "c"]), (["x", "y"], ["y", "x"]), (["p"], ["p"])]:
            d = token_distance(a, b)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_edit_distance_identity(self):
        self.assertEqual(edit_distance({"ADD_If"}, {"ADD_If"}), 0.0)

    def test_edit_distance_disjoint(self):
        self.assertEqual(edit_distance({"ADD_If"}, {"ADD_For"}), 1.0)

    def test_edit_distance_symmetric(self):
        a = {"ADD_If", "DEL_Name"}
        b = {"ADD_If", "ADD_Call"}
        self.assertEqual(edit_distance(a, b), edit_distance(b, a))

    def test_edit_distance_empty_empty(self):
        self.assertEqual(edit_distance(set(), set()), 0.0)

    def test_edit_distance_bounds(self):
        cases = [({"A"}, {"B", "C"}), ({"X", "Y"}, {"Y"}), ({"P"}, {"P"})]
        for a, b in cases:
            d = edit_distance(a, b)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_module_distance_jaccard(self):
        a = {"django/db", "django/core"}
        b = {"django/db", "django/forms"}
        self.assertAlmostEqual(module_distance(a, b), 1 - 1 / 3)

    def test_module_distance_empty(self):
        self.assertEqual(module_distance(set(), set()), 0.0)


class TestDiff(unittest.TestCase):
    def test_identity_zero_distance_all_levels(self):
        v = _sample_view(
            edits=["ADD_If", "DEL_Name"],
            tokens=["ADD_If", "DEL_Name"],
            scopes=["FunctionDef:foo"],
            modules=["django/db"],
        )
        d = diff(v, v)
        self.assertEqual(d.tokens, 0.0)
        self.assertEqual(d.edits, 0.0)
        self.assertEqual(d.scopes, 0.0)
        self.assertEqual(d.modules, 0.0)

    def test_symmetry(self):
        va = _sample_view(
            edits=["ADD_If", "ADD_Call"],
            tokens=["ADD_If", "ADD_Call"],
            scopes=["FunctionDef:foo"],
            modules=["django/db"],
        )
        vb = _sample_view(
            edits=["ADD_If", "DEL_Name", "ADD_Return"],
            tokens=["DEL_Name", "ADD_Return"],
            scopes=["FunctionDef:bar"],
            modules=["django/core"],
        )
        d1 = diff(va, vb)
        d2 = diff(vb, va)
        self.assertAlmostEqual(d1.tokens, d2.tokens)
        self.assertAlmostEqual(d1.edits, d2.edits)
        self.assertAlmostEqual(d1.scopes, d2.scopes)
        self.assertAlmostEqual(d1.modules, d2.modules)

    def test_bounds(self):
        va = _sample_view(
            edits=["ADD_If"],
            tokens=["ADD_If"],
            scopes=["FunctionDef:f"],
            modules=["a/b"],
        )
        vb = _sample_view(
            edits=["ADD_For"],
            tokens=["ADD_For"],
            scopes=["FunctionDef:g"],
            modules=["c/d"],
        )
        d = diff(va, vb)
        for lv in ("tokens", "edits", "scopes", "modules"):
            val = getattr(d, lv)
            self.assertIsNotNone(val)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_empty_inputs_do_not_crash(self):
        va = _sample_view(edits=[], tokens=[], scopes=[], modules=[])
        vb = _sample_view(edits=[], tokens=[], scopes=[], modules=[])
        d = diff(va, vb)
        self.assertIsInstance(d, Diff)
        # All levels unavailable -> all None
        self.assertEqual(d.available_levels, [])

    def test_partial_availability(self):
        va = _sample_view(edits=["ADD_If"], tokens=[], scopes=[], modules=["x/y"])
        vb = _sample_view(edits=["ADD_For"], tokens=[], scopes=[], modules=["x/y"])
        d = diff(va, vb)
        self.assertIsNotNone(d.edits)
        self.assertIsNotNone(d.modules)
        self.assertIsNone(d.tokens)
        self.assertIsNone(d.scopes)

    def test_mean_of_available_levels(self):
        va = _sample_view(edits=["A"], tokens=[], scopes=[], modules=["x"])
        vb = _sample_view(edits=["B"], tokens=[], scopes=[], modules=["x"])
        d = diff(va, vb)
        # edits=1.0 (disjoint), modules=0.0 (same) → mean 0.5
        self.assertAlmostEqual(d.mean(), 0.5)


class TestSignatureAndOOD(unittest.TestCase):
    def test_signature_counts(self):
        population = [
            _sample_view(edits=["ADD_If"], tokens=[], scopes=[], modules=["a/b"]),
            _sample_view(edits=["ADD_If", "ADD_For"], tokens=[], scopes=[], modules=["a/b"]),
            _sample_view(edits=["ADD_Return"], tokens=[], scopes=[], modules=["c/d"]),
        ]
        sig = signature(population)
        self.assertEqual(sig.n, 3)
        self.assertEqual(sig.edit_vocab, frozenset({"ADD_If", "ADD_For", "ADD_Return"}))
        self.assertEqual(sig.edit_freq["ADD_If"], 2)
        self.assertEqual(sig.module_vocab, frozenset({"a/b", "c/d"}))

    def test_ood_score_zero_when_fully_covered(self):
        population = [_sample_view(edits=["ADD_If", "ADD_For"], tokens=[], scopes=[], modules=[])]
        ref = build_reference_vocabulary(population)
        t = _sample_view(edits=["ADD_If"], tokens=[], scopes=[], modules=[])
        self.assertEqual(ood_score(t, ref, level="edits"), 0.0)

    def test_ood_score_one_when_fully_novel(self):
        population = [_sample_view(edits=["ADD_If"], tokens=[], scopes=[], modules=[])]
        ref = build_reference_vocabulary(population)
        t = _sample_view(edits=["ADD_For"], tokens=[], scopes=[], modules=[])
        self.assertEqual(ood_score(t, ref, level="edits"), 1.0)

    def test_ood_score_partial(self):
        population = [_sample_view(edits=["ADD_If", "ADD_Return"], tokens=[], scopes=[], modules=[])]
        ref = build_reference_vocabulary(population)
        t = _sample_view(edits=["ADD_If", "ADD_For", "ADD_Call"], tokens=[], scopes=[], modules=[])
        self.assertAlmostEqual(ood_score(t, ref, level="edits"), 2 / 3)

    def test_ood_items_lists_novel(self):
        population = [_sample_view(edits=["ADD_If"], tokens=[], scopes=[], modules=[])]
        ref = build_reference_vocabulary(population)
        t = _sample_view(edits=["ADD_If", "ADD_For"], tokens=[], scopes=[], modules=[])
        self.assertEqual(ood_items(t, ref, level="edits"), ["ADD_For"])

    def test_ood_empty_trajectory(self):
        population = [_sample_view(edits=["ADD_If"], tokens=[], scopes=[], modules=[])]
        ref = build_reference_vocabulary(population)
        t = _sample_view(edits=[], tokens=[], scopes=[], modules=[])
        self.assertEqual(ood_score(t, ref, level="edits"), 0.0)


class TestRealDataRoundtrip(unittest.TestCase):
    """One end-to-end check: load two real traces, compute their diff."""

    @unittest.skipUnless(RESOLVED_LITE.exists(), f"resolved traces not found at {RESOLVED_LITE}")
    def test_two_real_traces_produce_wellformed_diff(self):
        traces = []
        with open(RESOLVED_LITE) as fh:
            for line in fh:
                traces.append(json.loads(line))
                if len(traces) == 2:
                    break

        self.assertEqual(len(traces), 2)
        va = view_from_trace(traces[0])
        vb = view_from_trace(traces[1])

        # At least one of edits / modules should be available
        self.assertTrue(va.has_edits or va.has_modules)
        self.assertTrue(vb.has_edits or vb.has_modules)

        d = diff(va, vb)
        for lv in d.available_levels:
            val = getattr(d, lv)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    @unittest.skipUnless(RESOLVED_LITE.exists(), f"resolved traces not found at {RESOLVED_LITE}")
    def test_view_from_patch(self):
        patch = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,3 +1,5 @@\n"
            " def f(x):\n"
            "+    if x is None:\n"
            "+        return None\n"
            "     return x + 1\n"
        )
        v = view_from_patch(patch, file_paths=["foo"], scopes=["FunctionDef:f"])
        self.assertTrue(v.has_edits)
        self.assertIn("foo", v.modules)


if __name__ == "__main__":
    unittest.main()
