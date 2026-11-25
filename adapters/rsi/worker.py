"""Trusted evaluation entry point, executed only inside the Linux RSI sandbox.

The mounted source contains questions without gold/metadata. Real gold and the
optimizer's records are never mounted. Synthetic preflight keeps its own gold.
"""
from __future__ import annotations

import argparse
import errno
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiment'))
from common import atomic_json, read_json, rows, BuildClock
from feedback import sha
from experiment import Campaign, Dataset, bounded_pool, qa_one
from rise.retrieval import build_index, load_index


def link_or_copy(source, target):
    # Linux cannot hardlink between separate bind mounts, even on one disk.
    # Preserve bytes and exclusive creation; never make the corpus writable.
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EROFS):
            raise
        with open(source, 'rb') as src, open(target, 'xb') as dst:
            shutil.copyfileobj(src, dst)


def prepare(self):
    """Reuse the host-verified read-only corpus, not 500k copied files per phase."""
    self.data = self.source
    self.documents = rows(self.source / 'documents.jsonl')
    self.questions = rows(self.source / 'qa.jsonl')
    if len(self.questions) != self.spec['expected_qa_count']:
        raise ValueError('Question count mismatch')
    if any(set(q) != {'id', 'question', 'gold_answers'} or q['gold_answers'] for q in self.questions):
        raise ValueError('Only sanitized questions may enter QA evaluation')
    self.source_document_count = len(self.documents)
    self.path_map = {d['id']: str(Path(d['path']).relative_to('corpus')) for d in self.documents}
    atomic_json(self.root / 'document_map.json', self.path_map)
    self.status('prepared')


def build(self):
    # /cache belongs to one frozen candidate. It is shared only between that
    # candidate's dev/selection/test phases, not between different source trees.
    target = Path('/cache/index')
    receipt = target / 'COMPLETE.json'
    identity = dict(code=self.campaign.config['code'], documents=sha(self.source / 'documents.jsonl'))
    if receipt.exists():
        recorded = read_json(receipt)
        if recorded['identity'] != identity:
            raise ValueError('Cached index belongs to another code/data version')
        for rel, expected in recorded['files'].items():
            if sha(target / rel) != expected:
                raise ValueError('Cached index changed')
        return
    staging = Path('/cache/index.building')
    if staging.exists() or target.exists():
        raise ValueError('Incomplete index retained; inspect before restarting')
    clock = BuildClock('/cache/build_clock')
    try:
        build_index(Path('/corpus.parquet'), staging, k1=1.5, b=0.75)
        files = {str(p.relative_to(staging)): sha(p) for p in staging.rglob('*') if p.is_file()}
        atomic_json(staging / 'COMPLETE.json', dict(identity=identity, files=files))
        staging.rename(target)
        clock.finish('complete')
    except BaseException:
        clock.finish('failed')
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--stage', choices=['preflight', 'qa'], required=True)
    args = parser.parse_args()
    if not Path('/rsi-sandbox').is_file():
        raise RuntimeError('Run via adapters/rsi/run.py, never directly on the host')
    import rise.tools as tools
    tools.os = SimpleNamespace(**vars(os))
    tools.os.link = link_or_copy
    Dataset.prepare = prepare
    if args.stage == 'preflight':
        # Keep the existing real API/tool probes, with only preparation adapted.
        import preflight
        sys.argv = [sys.argv[0], '--plan', str(args.plan)]
        preflight.main()
        return
    c = Campaign(args.plan)
    receipt = read_json(c.root / 'preflight_passed.json')
    if receipt['signature'] != c.signature:
        raise ValueError('Preflight signature mismatch')
    for ds in c.datasets:
        ds.prepare()
        build(ds)
        ds.searcher, ds.doc_ids = load_index('/cache/index')
        jobs = [(ds, q) for q in ds.questions if not (ds.qaroot(q) / 'result.json').exists()]
        bounded_pool(jobs, qa_one, workers=c.concurrency['qa'], activity=c.activity['qa'])
        if not all(ds.qa_valid(q) for q in ds.questions):
            raise RuntimeError('Incomplete QA; failures retained, not rerolled')
    c.status('qa_complete')


if __name__ == '__main__':
    main()
