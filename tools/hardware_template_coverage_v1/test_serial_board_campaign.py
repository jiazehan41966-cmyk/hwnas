"""A transport recovery must never convert an unfinished measurement into PASS."""
import json
from pathlib import Path
import tempfile
import unittest

from .serial_board_campaign import terminal_fetch_recovery_candidate


class TerminalSnapshotRecoveryGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='t6_queue_recovery_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.row = dict(case_id='case', job_prefix='job')
        execution = self.root / 'board_execution/job'
        execution.mkdir(parents=True)
        self.events = execution / 'events.jsonl'
        self.original = self.root / 'board_snapshots/job_timing'
        self.original.mkdir(parents=True)
        self.write('PASS')

    def write(self, state, reason='bridge response timed out; inspect status before any launch retry',
              case='case'):
        self.events.write_text('\n'.join(json.dumps(row) for row in [
            dict(stage='timing_state', case_id=case, state=state),
            dict(stage='case_stopped_preserving_evidence', case_id=case,
                 error_type='RuntimeError', reason=reason)]) + '\n', encoding='utf-8')

    def test_completed_pass_with_empty_download_can_recover(self):
        self.assertTrue(terminal_fetch_recovery_candidate(self.root, self.row))

    def test_running_failed_or_unknown_state_cannot_recover(self):
        for state in ('RUNNING', 'FAIL', 'DEVICE_STATE_UNKNOWN', 'DMA_QUARANTINED'):
            with self.subTest(state=state):
                self.write(state)
                self.assertFalse(terminal_fetch_recovery_candidate(self.root, self.row))

    def test_other_error_or_case_binding_cannot_recover(self):
        self.write('PASS', reason='bridge returned no unique safe receipt')
        self.assertFalse(terminal_fetch_recovery_candidate(self.root, self.row))
        self.write('PASS', case='other')
        self.assertFalse(terminal_fetch_recovery_candidate(self.root, self.row))

    def test_partial_download_or_existing_recovery_cannot_be_overwritten(self):
        part = self.original / 'snapshot.tar.gz'
        part.write_bytes(b'partial')
        self.assertFalse(terminal_fetch_recovery_candidate(self.root, self.row))
        part.unlink()
        (self.root / 'board_snapshots/job_timing_recovered').mkdir()
        self.assertFalse(terminal_fetch_recovery_candidate(self.root, self.row))

    def test_already_selected_case_is_not_reimported(self):
        pointer = self.root / 'cases/case/board/selected_run.json'
        pointer.parent.mkdir(parents=True)
        pointer.write_text('{}')
        self.assertFalse(terminal_fetch_recovery_candidate(self.root, self.row))


if __name__ == '__main__':
    unittest.main()
