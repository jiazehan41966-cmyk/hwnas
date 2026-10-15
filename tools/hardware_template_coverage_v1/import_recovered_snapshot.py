"""Apply the unchanged timing validator/importer to a recovered terminal snapshot."""
import argparse
import json
from pathlib import Path
import sys
from .run_board_case import import_timing, save_new
from .board_transfer import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        raise ValueError('run name must be one path component')
    result = import_timing(args.snapshot, args.package, args.attempt, args.run_name)
    save_new(args.snapshot / 'import_recovery.json', dict(result=result, argv=sys.argv,
        importer_sha256=sha(__file__), cwd=str(Path.cwd()), measurement_restarted=False))
    print(json.dumps({key:result[key] for key in ('status','case_id','numerical_status',
        'hardware_measurement_eligible','protocol_conditions_met')}))


if __name__ == '__main__':
    main()
