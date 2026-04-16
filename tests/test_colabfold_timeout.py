"""Tests for Fix 10: ColabFold subprocess timeout.

_run_colabfold_subprocess must:
- Pass a timeout to proc.wait()
- Kill the process and raise RuntimeError on timeout
- Accept a configurable timeout parameter
"""
import subprocess
import pytest
from unittest.mock import MagicMock, patch, call


class TestColabFoldSubprocessTimeout:
    def _make_mock_proc(self, stdout_lines=None, returncode=0):
        proc = MagicMock()
        proc.stdout = iter(stdout_lines or [])
        proc.returncode = returncode
        return proc

    def test_proc_wait_called_with_timeout(self):
        """Fix 10: proc.wait() must be called with a timeout argument."""
        from ghostfold.core.colabfold import _run_colabfold_subprocess

        mock_proc = self._make_mock_proc()

        with patch("ghostfold.core.colabfold.subprocess.Popen",
                   return_value=mock_proc):
            _run_colabfold_subprocess(
                gpu_id=0,
                msa_file="/tmp/test.a3m",
                output_dir="/tmp/out",
                max_seq=32,
                max_extra_seq=64,
                launcher_prefix=("colabfold_batch",),
                launcher_cwd=None,
                cache_home=None,
            )

        # proc.wait must have been called with a timeout keyword argument
        mock_proc.wait.assert_called_once()
        _, kwargs = mock_proc.wait.call_args
        assert "timeout" in kwargs, (
            "proc.wait() must be called with timeout= to prevent infinite hangs"
        )
        assert kwargs["timeout"] > 0

    def test_timeout_kills_process_and_raises(self):
        """Fix 10: TimeoutExpired → kill process → raise RuntimeError."""
        from ghostfold.core.colabfold import _run_colabfold_subprocess

        mock_proc = self._make_mock_proc()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="colabfold_batch", timeout=3600)

        with patch("ghostfold.core.colabfold.subprocess.Popen",
                   return_value=mock_proc):
            with pytest.raises(RuntimeError, match="[Tt]imed? ?out"):
                _run_colabfold_subprocess(
                    gpu_id=0,
                    msa_file="/tmp/test.a3m",
                    output_dir="/tmp/out",
                    max_seq=32,
                    max_extra_seq=64,
                    launcher_prefix=("colabfold_batch",),
                    launcher_cwd=None,
                    cache_home=None,
                )

        # Process must be killed on timeout
        mock_proc.kill.assert_called_once()

    def test_default_timeout_is_sensible(self):
        """Fix 10: default timeout must be >= 1 hour (3600s)."""
        from ghostfold.core.colabfold import _run_colabfold_subprocess
        import inspect

        sig = inspect.signature(_run_colabfold_subprocess)
        if "timeout" in sig.parameters:
            default = sig.parameters["timeout"].default
            assert default != inspect.Parameter.empty, "timeout must have a default"
            assert default >= 3600, f"Default timeout {default}s < 3600s (1 hour)"

    def test_non_timeout_error_still_raises_calledprocesserror(self):
        """Fix 10: Non-zero returncode (not timeout) must still raise CalledProcessError."""
        from ghostfold.core.colabfold import _run_colabfold_subprocess

        mock_proc = self._make_mock_proc(returncode=1)

        with patch("ghostfold.core.colabfold.subprocess.Popen",
                   return_value=mock_proc):
            with pytest.raises(subprocess.CalledProcessError):
                _run_colabfold_subprocess(
                    gpu_id=0,
                    msa_file="/tmp/test.a3m",
                    output_dir="/tmp/out",
                    max_seq=32,
                    max_extra_seq=64,
                    launcher_prefix=("colabfold_batch",),
                    launcher_cwd=None,
                    cache_home=None,
                )
