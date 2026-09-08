"""The qualification harness must fail closed when test execution is ambiguous."""
import unittest

from check_numerical_mutations import (
    GROUPS, MUTATIONS, Mutation, evidence_log_name, replace_source, validate_result,
)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.failure = (
            "running 1 test\ntest oracle ... FAILED\n\n"
            "thread 'oracle' panicked at src/oracle.rs:1:1:\noptimal cost: wrong\n"
            "\nfailures:\n    oracle\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured;\n"
        )

    def check_failure(self, code, output):
        validate_result(code, output, 1, "oracle", ("optimal cost:",))

    def test_intended_assertion_is_accepted(self):
        self.check_failure(101, self.failure)

    def test_compile_error_is_not_evidence(self):
        with self.assertRaises(RuntimeError):
            self.check_failure(101, "error[E0308]: mismatched types")

    def test_zero_discovered_tests_are_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check_failure(0, "running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored;")

    def test_ignored_test_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check_failure(0, "running 1 test\ntest oracle ... ignored\n"
                                  "test result: ok. 0 passed; 0 failed; 1 ignored;")

    def test_unrelated_panic_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check_failure(101, self.failure.replace("optimal cost: wrong", "index out of bounds"))

    def test_wrong_test_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check_failure(101, self.failure.replace("test oracle ...", "test other ..."))

    def test_missing_summary_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check_failure(101, self.failure.partition("test result:")[0])

    def test_wrong_exit_status_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check_failure(1, self.failure)

    def test_success_and_survivor(self):
        success = "running 1 test\ntest oracle ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored;"
        validate_result(0, success, 1)
        with self.assertRaises(RuntimeError):
            self.check_failure(0, success)

    def test_evidence_filenames_are_portable_and_unique(self):
        labels = [m.label for m in MUTATIONS]
        labels += [stage + (test or target) for stage in ("before-", "restored-")
                   for target, test, _ in GROUPS]
        names = [evidence_log_name(label) for label in labels]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertRegex(name, r"^[A-Za-z0-9_.-]+\.log$")
        self.assertEqual(evidence_log_name("a::b"), "a--b.log")

    def test_source_anchor_cardinality(self):
        mutation = Mutation("m", "source", "old", "new", "lib", "oracle", ("cost",), matches=2)
        self.assertEqual(replace_source("old old", mutation), "new old")
        with self.assertRaises(RuntimeError):
            replace_source("old", mutation)


if __name__ == "__main__":
    unittest.main()
