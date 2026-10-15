"""Replay the prior independent local checker against this reboot's inputs.

The checked algorithm stays byte-identical. Only explicitly recorded input paths
and the lease digest change. No remote operations or new independent review claim.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/hardware_template_coverage_20261013/attempt01'
CHECKER = BASE / 'board_integration_reviews/ownership_board_evidence_v1/check_local_evidence.py'
CHECKER_SHA = '51132fcd88ad78cda2062bac08f2702ef88e49b6e8191afc9486150d7eae4d55'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--ownership-config', required=True, type=Path)
    parser.add_argument('--reserve-receipt', required=True, type=Path)
    parser.add_argument('--snapshot', action='append', default=[], type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert sha(CHECKER) == CHECKER_SHA, 'independent checker source changed'
    receipt, config = load(args.reserve_receipt), load(args.ownership_config)
    assert receipt['ok'] is True and receipt['status'] == 'NEW_BOOT_RESERVED_AND_CHECKED'
    assert receipt['config'] == config
    args.output.mkdir(parents=True, exist_ok=False)
    extracted = args.output / 'lease_receipt_extracted.json'
    save(extracted, receipt['receipt'])
    save(args.output / 'inputs.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        wrapper_sha256=sha(__file__), checker_path=str(CHECKER), checker_sha256=CHECKER_SHA,
        index_sha256=sha(args.index), ownership_config_sha256=sha(args.ownership_config),
        original_reserve_receipt_sha256=sha(args.reserve_receipt),
        extracted_receipt_sha256=sha(extracted), receipt_extraction='original.receipt',
        scope='ROOT_REPLAY_OF_PREVIOUS_INDEPENDENT_CHECKER_WITH_NEW_BOOT_INPUTS'))
    namespace = dict(__name__='bound_independent_checker', __file__=str(CHECKER))
    exec(compile(CHECKER.read_bytes(), str(CHECKER), 'exec'), namespace)
    namespace.update(INDEX=args.index.resolve(), LEASE_CONFIG=args.ownership_config.resolve(),
                     LEASE_RECEIPT=extracted.resolve(), EXPECTED_LEASE=config['reservation']['sha256'])
    packet = namespace['packet_review']()
    save(args.output / 'packet_review.json', packet)
    results = []
    for snapshot in args.snapshot:
        checked = namespace['snapshot_review'](snapshot, packet)
        save(args.output / (checked['job_id'] + '.json'), checked)
        results.append(dict(job_id=checked['job_id'], status=checked['status'], mode=checked['mode']))
    result = dict(status='PASS_REPLAYED_PACKET_AND_REQUESTED_SNAPSHOT_BINDINGS', results=results,
                  independently_authored_checker=True, new_independent_human_or_agent_review=False)
    save(args.output / 'summary.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
