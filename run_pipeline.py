#!/usr/bin/env python3
"""Single offline entry point for Qn EGO HDF5 hand processing."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from qn_ego.pipeline import EmfDataLossError, main as _main


def main(argv=None):
    """Programmatic and CLI entry point.

    Passing a list uses the same parser as the command line without touching
    the embedding application's ``sys.argv``.
    """

    try:
        return _main(argv)
    except EmfDataLossError as exc:
        # The complete child traceback remains in RUN/calls/*.log.  Keep the
        # platform/CLI surface short and actionable for the operator.
        detail = str(exc).removeprefix('EMF_DATA_LOSS: ').strip()
        print('ERROR: EMF数值丢失；' + detail, file=sys.stderr, flush=True)
        if argv is not None:
            raise SystemExit(2)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
