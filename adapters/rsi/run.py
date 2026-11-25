#!/usr/bin/env python3
"""Small RSI controller: frozen baseline → query/edit → paired evaluation.

init/query are offline. run/test require --execute and Linux bubblewrap.
Candidate Python is never imported into the controller process.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from feedback import Feedback, audit_usage, checked_results, comparison, events, page, sha
from common import ROOT, MODEL, RESPONSE_MODEL, atomic_json, append, code_manifest, digest, now, read_json, safe_id
from prompts import SYSTEM, TOOLS

EDITABLE = ['src/rise/agent.py', 'src/rise/tools.py', 'src/rise/rise_agent.py', 'src/rise/retrieval.py']


class BudgetStop(RuntimeError):
    pass


def manifest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.txt'}}


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen record changed: ' + str(path))
    else:
        atomic_json(path, value)


def validate_split(questions, split):
    ids = [qid for name in ('dev', 'selection', 'test') for qid in split[name]]
    if any(not split[name] for name in ('dev', 'selection', 'test')):
        raise ValueError('All three splits must be nonempty')
    if len(ids) != len(set(ids)) or set(ids) != {q['id'] for q in questions}:
        raise ValueError('Splits must partition the baseline question set exactly')
    assignment = {qid: name for name in ('dev', 'selection', 'test') for qid in split[name]}
    owners = {}
    for q in questions:
        keys = [('question', ' '.join(q['question'].lower().split()))]
        keys += [('document', d) for d in q.get('document_ids', [])]
        if q['id'] in split.get('groups', {}):
            keys.append(('event', split['groups'][q['id']]))
        for key in keys:
            if key in owners and owners[key] != assignment[q['id']]:
                raise ValueError('Related questions cross split boundary: ' + str(key))
            owners[key] = assignment[q['id']]


def apply_edits(parent, destination, edits):
    """Stage an isolated candidate; syntax-check without importing generated code."""
    if not edits:
        raise ValueError('At least one edit is required')
    staged = {}
    for edit in edits:
        name = edit['path']
        if name not in EDITABLE:
            raise ValueError('Not an editable runtime module: ' + name)
        text = staged.get(name, (Path(parent) / name).read_text())
        if not edit['old'] or text.count(edit['old']) != 1:
            raise ValueError('Old text must match exactly once: ' + name)
        text = text.replace(edit['old'], edit['new'], 1)
        compile(text, name, 'exec')
        staged[name] = text
    if all(text == (Path(parent) / name).read_text() for name, text in staged.items()):
        raise ValueError('Candidate has no source change')
    temp = Path(str(destination) + '.building')
    shutil.copytree(parent, temp)
    for name, text in staged.items():
        (temp / name).write_text(text)
    temp.rename(destination)


def initialize(args):
    plan = read_json(args.plan)
    if len(plan['datasets']) != 1 or plan.get('structured_docs') is not False:
        raise ValueError('RSI starts from one no-TOC RISE-BM25 campaign')
    spec = plan['datasets'][0]
    if spec.get('excluded_document_ids'):
        raise ValueError('Use the fixed full corpus, without excluded documents')
    baseline = Path(plan['run_root']).resolve()
    effective = read_json(baseline / 'effective_config.json')
    if effective['code'] != code_manifest():
        raise ValueError('Baseline source differs from current code; use its exact checkout')
    if (effective['datasets'] != plan['datasets'] or effective['algorithm']['structured_docs']
            or effective['algorithm']['coerce_final'] != plan['coerce_final']
            or any(effective['concurrency'][k] != v for k, v in plan['concurrency'].items())):
        raise ValueError('Plan does not describe the recorded baseline')
    source = Path(spec['source']).resolve()
    saved_data = read_json(baseline / spec['name'] / 'config.json')['data']
    if saved_data['documents'] != sha(source / 'documents.jsonl') or saved_data['qa'] != sha(source / 'qa.jsonl'):
        raise ValueError('Prepared data differs from the recorded baseline')
    questions = [q for _, q in events(source / 'qa.jsonl') if q['id'] in spec['qa_ids']]
    split = read_json(args.split)
    validate_split(questions, split)
    checked_results(baseline / spec['name'], spec['qa_ids'], digest(effective))
    parquet = baseline / spec['name'] / 'corpus.parquet'
    config = dict(schema=1, plan=plan, baseline=str(baseline), source=str(source),
        dataset=spec['name'], baseline_signature=digest(effective), split=split,
        data_hashes={name: sha(source / name) for name in ('documents.jsonl', 'qa.jsonl')},
        parquet=str(parquet), parquet_sha256=sha(parquet),
        optimizer=dict(model=args.optimizer_model, response_model=args.optimizer_response_model,
                       max_output_tokens=8192, temperature=0, max_calls=args.max_calls),
        max_candidates=args.max_candidates, patience=2, max_total_tokens=args.max_total_tokens,
        quality_tolerance=args.quality_tolerance, min_saving=args.min_saving,
        implementation=manifest(Path(__file__).parent))
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    code = root / 'versions/baseline/code'
    for rel in list(code_manifest()) + ['adapters/rsi/' + name for name in config['implementation']]:
        target = code / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    (code / '.venv').mkdir()  # Mount point beneath the read-only code tree.
    config['baseline_code'] = manifest(code)
    config['signature'] = digest(config)
    atomic_json(root / 'config.json', config)
    atomic_json(root / 'state.json', dict(current='baseline', completed=[], stalls=0, calls=0,
        stopped=False, messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content':
        json.dumps(dict(editable=EDITABLE, dev_questions=len(split['dev']),
                        limits={k: config[k] for k in ('max_candidates', 'quality_tolerance', 'min_saving')},
                        task='Inspect the baseline and propose a justified improvement.'), ensure_ascii=False)}]))
    print('Initialized', root, '(no API calls; baseline results are reused)')


class Controller:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = (self.root / 'controller.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.config = read_json(self.root / 'config.json')
        c = self.config
        if c['signature'] != digest({k: v for k, v in c.items() if k != 'signature'}):
            raise ValueError('RSI configuration changed')
        if c['implementation'] != manifest(Path(__file__).parent) or c['baseline_code'] != manifest(self.code('baseline')):
            raise ValueError('Controller or frozen baseline changed; start a new RSI campaign')
        baseline = read_json(Path(c['baseline']) / 'effective_config.json')
        if digest(baseline) != c['baseline_signature'] or baseline['code'] != code_manifest():
            raise ValueError('Host scorer/runtime source changed')
        for name, expected in c['data_hashes'].items():
            if sha(Path(c['source']) / name) != expected:
                raise ValueError('Frozen data manifest changed')
        self.questions = {q['id']: q for _, q in events(Path(c['source']) / 'qa.jsonl')}
        self.state = read_json(self.root / 'state.json')
        self._feedback = None
        if self.state['current'] != 'baseline':
            if manifest(self.code(self.state['current'])) != read_json(self.code(self.state['current']).parent / 'code.json'):
                raise ValueError('Accepted candidate source changed')

    def save(self):
        atomic_json(self.root / 'state.json', self.state)

    def code(self, version):
        return self.root / 'versions' / version / 'code'

    def location(self, version, split='dev'):
        if version == 'baseline':
            return Path(self.config['baseline']) / self.config['dataset'], self.config['baseline_signature']
        run = self.root / 'evaluations' / version / split / 'run'
        return run / self.config['dataset'], digest(read_json(run / 'effective_config.json'))

    def feedback(self, version=None):
        version = version or self.state['current']
        if version not in ['baseline', *self.state['completed']]:
            raise ValueError('Use baseline or a returned completed candidate version')
        if self._feedback is not None and self._feedback[0] == version:
            return self._feedback[1]
        dataset, signature = self.location(version)
        scoring = None if version == 'baseline' else self.root / 'evaluations' / version / 'dev' / 'scoring' / self.config['dataset']
        feedback = Feedback(dataset, self.config['source'],
            [self.questions[k] for k in self.config['split']['dev']], signature,
            self.root / 'annotations' / version, scoring=scoring)
        self._feedback = (version, feedback)
        return feedback

    def results(self, version, split):
        dataset, signature = self.location(version, split)
        if version != 'baseline':
            dataset = self.root / 'evaluations' / version / split / 'scoring' / self.config['dataset']
        return checked_results(dataset, self.config['split'][split], signature)

    def diagnosis(self, ident):
        paths = [p for p in (self.root / 'diagnoses').glob('*.json') if p.stem == ident]
        if not paths:
            raise ValueError('Unknown diagnosis_id; use diagnose first')
        record = read_json(paths[0])
        if digest(record)[:24] != ident:
            raise ValueError('Frozen diagnosis changed')
        return record

    def query(self, kind, qid=None, ref=None, offset=0, limit=6000, version=None):
        if kind == 'diagnoses':
            if ref:
                return page(self.diagnosis(ref), offset, limit)
            outcomes = {}
            for path in (self.root / 'versions').glob('*/proposal.json'):
                proposal = read_json(path)
                decision = path.parent / 'decision.json'
                outcomes.setdefault(proposal['proposal']['diagnosis_id'], []).append(
                    read_json(decision) if decision.exists() else dict(version=path.parent.name, status='pending'))
            records = []
            for path in sorted((self.root / 'diagnoses').glob('*.json')):
                d = self.diagnosis(path.stem)
                records.append(dict(diagnosis_id=path.stem, version=d['version'],
                    needs_regrounding=d['version'] != self.state['current'], hypothesis=d['claim']['hypothesis'],
                    expected_change=d['claim']['expected_change'], risks=d['claim']['risks'], metrics=d['metrics'],
                    necessity_judgments=[dict(qid=c['qid'], necessity=c['necessity'], reason=c['reason'],
                                             matched_span_ids=c['cost_scope']['matched_span_ids']) for c in d['cases']],
                    outcomes=outcomes.get(path.stem, [])))
            value = dict(current_version=self.state['current'],
                pilot_qa_tokens_reference=sum(v['tokens'] for v in self.results(self.state['current'], 'dev').values()),
                validation_note='Common dev QA reference, not a forecast. Add preflight/judge and inspect prior actual evaluation_usage.',
                budget=self.budget(), diagnoses=records)
        elif kind == 'code':
            if ref not in self.config['baseline_code']:
                raise ValueError('Unknown source file')
            if (version or self.state['current']) not in ['baseline', *self.state['completed']]:
                raise ValueError('Unknown candidate version')
            value = (self.code(version or self.state['current']) / ref).read_text()
        else:
            return self.feedback(version).query(kind, qid, ref, offset, limit)
        return page(value, offset, limit)

    def budget(self):
        usage = audit_usage(self.root / 'evaluations')
        optimizer = audit_usage(self.root / 'optimizer')
        report = dict(evaluation=usage, optimizer=optimizer,
                      total_tokens=usage['total_tokens'] + optimizer['total_tokens'],
                      usage_complete=usage['usage_complete'] and optimizer['usage_complete'],
                      baseline_reused=True, baseline_cost_excluded=True,
                      policy='Admission checked between optimizer requests and evaluation stages; one stage can exceed the token threshold')
        atomic_json(self.root / 'budget.json', report)
        return report

    def admit(self):
        budget = self.budget()
        if not budget['usage_complete']:
            raise RuntimeError('Unknown request cost; inspect HTTP audit before continuing')
        if budget['total_tokens'] >= self.config['max_total_tokens']:
            raise BudgetStop('RSI token admission budget reached')

    def sandbox_command(self, version, phase, stage):
        c = self.config
        command = [shutil.which('bwrap'), '--die-with-parent', '--unshare-all', '--share-net', '--clearenv']
        for name in ('/usr', '/bin', '/sbin', '/lib', '/lib64'):
            p = Path(name)
            if p.is_symlink():
                command += ['--symlink', str(p.readlink()), name]
            elif p.exists():
                command += ['--ro-bind', name, name]
        # A venv may depend on a separately installed Python outside /usr.
        base = Path(sys.base_prefix).resolve()
        if not base.is_relative_to('/usr'):
            command += ['--ro-bind', str(base), str(base)]
        command += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/etc']
        for name in ('/etc/ssl', '/etc/ld.so.cache', '/etc/resolv.conf', '/etc/hosts'):
            if Path(name).exists():
                command += ['--ro-bind', name, name]
        cache = self.root / 'cache' / version
        cache.mkdir(parents=True, exist_ok=True)
        mounts = [(self.code(version), '/app', False), (ROOT / '.venv', '/app/.venv', False),
                  (phase / 'input', '/source', False), (Path(c['source']) / 'corpus', '/source/corpus', False),
                  (c['parquet'], '/corpus.parquet', False), (phase / 'run', '/run', True),
                  (cache, '/cache', True), (phase / 'plan.json', '/plan.json', False),
                  (c['plan']['credentials'], '/credentials.json', False),
                  (c['plan']['rg_binary'], '/rg', False), (phase / 'marker', '/rsi-sandbox', False)]
        for source, target, writable in mounts:
            command += ['--bind' if writable else '--ro-bind', str(source), target]
        for key, value in dict(PATH='/usr/bin:/bin', HOME='/tmp', LANG='C.UTF-8',
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1').items():
            command += ['--setenv', key, value]
        return command + ['--chdir', '/app', '--', '/app/.venv/bin/python', '-B',
                          'adapters/rsi/worker.py', '--plan', '/plan.json', '--stage', stage]

    def evaluate(self, version, split):
        c = self.config
        phase = self.root / 'evaluations' / version / split
        receipt = phase / 'complete.json'
        code = manifest(self.code(version))
        identity = dict(rsi_signature=c['signature'], version=version, split=split, code=code)
        if receipt.exists():
            if read_json(receipt) != identity:
                raise ValueError('Evaluation receipt changed')
            return self.results(version, split)
        self.admit()
        phase.mkdir(parents=True, exist_ok=True)
        write_once(phase / 'identity.json', identity)
        source = phase / 'input'
        source.mkdir(exist_ok=True)
        (source / 'corpus').mkdir(exist_ok=True)
        (phase / 'run').mkdir(exist_ok=True)
        (phase / 'marker').touch(exist_ok=True)
        if not (source / 'documents.jsonl').exists():
            shutil.copy2(Path(c['source']) / 'documents.jsonl', source / 'documents.jsonl')
        if sha(source / 'documents.jsonl') != c['data_hashes']['documents.jsonl']:
            raise ValueError('Sanitized document manifest changed')
        ids = c['split'][split]
        sanitized = [dict(id=k, question=self.questions[k]['question'], gold_answers=[]) for k in ids]
        qa = ''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in sanitized)
        qp = source / 'qa.jsonl'
        if qp.exists() and qp.read_text() != qa:
            raise ValueError('Sanitized evaluation input changed')
        qp.write_text(qa)
        plan = dict(run_root='/run', credentials='/credentials.json', rg_binary='/rg',
                    structured_docs=False, coerce_final=c['plan']['coerce_final'],
                    concurrency=c['plan']['concurrency'], datasets=[dict(name=c['dataset'],
                    source='/source', expected_qa_count=len(ids), qa_ids=ids, selected_qa_count=len(ids))])
        write_once(phase / 'plan.json', plan)
        for stage in ('preflight', 'qa'):
            self.admit()
            started = time.monotonic()
            with (phase / 'worker.log').open('a') as log:
                result = subprocess.run(self.sandbox_command(version, phase, stage),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    timeout=(len(ids) + 12) * 7200, check=False)
            append(phase / 'stages.jsonl', dict(stage=stage, seconds=time.monotonic() - started,
                                               returncode=result.returncode, timestamp=now()))
            if result.returncode:
                raise ValueError('Candidate evaluation failed; see retained worker.log')
        # Trusted host code alone sees actual gold and performs scoring.
        from experiment import judge_one, bounded_pool
        from runtime import ServiceGate
        dataset, signature = self.location(version, split)
        credentials = read_json(c['plan']['credentials'])
        campaign = SimpleNamespace(signature=signature, credentials=credentials,
                                   gates={'judge': ServiceGate(c['plan']['concurrency']['judge'])})
        # Scoring traces contain gold. Keep them outside every worker mount,
        # including when resuming after a scoring failure.
        scoring = phase / 'scoring' / c['dataset']
        ds = SimpleNamespace(root=scoring, campaign=campaign,
                             qaroot=lambda q: scoring / 'qa' / safe_id(q['id']))
        self.admit()
        pending = []
        for k in ids:
            write_once(ds.qaroot(self.questions[k]) / 'result.json',
                       read_json(dataset / 'qa' / safe_id(k) / 'result.json'))
            evaluation = ds.qaroot(self.questions[k]) / 'evaluation.json'
            if not evaluation.exists():
                pending.append((ds, self.questions[k]))
        bounded_pool(pending, judge_one, workers=c['plan']['concurrency']['judge'])
        values = self.results(version, split)
        write_once(receipt, identity)
        self.budget()
        return values

    def submit(self, call_id, proposal):
        version = safe_id(call_id)
        directory = self.root / 'versions' / version
        record = directory / 'proposal.json'
        if record.exists():
            saved = read_json(record)
            if saved['proposal'] != proposal:
                raise ValueError('Tool call ID reused with different edits')
            parent = saved['parent']
        else:
            if len(list((self.root / 'versions').glob('*/proposal.json'))) >= self.config['max_candidates']:
                raise ValueError('Candidate limit reached')
            parent = self.state['current']
            feedback = self.feedback()
            diagnosis = self.diagnosis(proposal['diagnosis_id'])
            if diagnosis['version'] != parent or not proposal['selection_reason'].strip():
                raise ValueError('Ground diagnosis on the current version and explain the selection')
            for case in diagnosis['cases']:
                if digest(feedback.annotation(case['qid'])) != case['annotation_hash']:
                    raise ValueError('Evidence annotation changed; record the diagnosis again')
                for reference in case['verified_refs']:
                    if feedback.query('event', qid=case['qid'], ref=reference['ref'])['sha256'] != reference['event_hash']:
                        raise ValueError('Cited original event changed')
            write_once(record, dict(parent=parent, proposal=proposal))
        decision_path = directory / 'decision.json'
        if decision_path.exists():
            decision = read_json(decision_path)
        else:
            decision = dict(accepted=False, parent=parent, version=version)
            try:
                if not self.code(version).exists():
                    apply_edits(self.code(parent), self.code(version), proposal['edits'])
                write_once(directory / 'code.json', manifest(self.code(version)))
                for split in ('dev', 'selection'):
                    values = self.evaluate(version, split)
                    before = self.results(parent, split)
                    delta = comparison(self.results('baseline', split), before, values,
                                       self.config['quality_tolerance'], self.config['min_saving'])
                    if split == 'dev':
                        delta['paired_questions'] = [dict(qid=k, tokens_saved=before[k]['tokens'] - values[k]['tokens'],
                            score_before=before[k]['score'], score_after=values[k]['score']) for k in values]
                    decision[split] = delta
                    if not delta['accepted']:
                        break
                else:
                    decision['accepted'] = True
            except (ValueError, SyntaxError, KeyError, FileNotFoundError) as exc:
                decision['error'] = 'Incomplete evaluation or invalid implementation: ' + type(exc).__name__
                # Detailed failures can name held-out questions; only aggregate
                # rejection reaches the optimizer, while the owner keeps the audit.
                write_once(directory / 'failure.json', dict(error=str(exc)))
            decision['evaluation_usage'] = audit_usage(self.root / 'evaluations' / version)
            spent = decision['evaluation_usage']
            dev = decision.get('dev')
            decision['dev_tokens_saved_per_evaluation_token'] = (
                dev['total_tokens_saved_from_current'] / spent['total_tokens']
                if dev and spent['usage_complete'] and spent['total_tokens'] else None)
            write_once(decision_path, decision)
        if version not in self.state['completed']:
            self.state['completed'].append(version)
            if decision['accepted']:
                self.state.update(current=version, stalls=0)
            else:
                self.state['stalls'] += 1
            self.save()
        return decision

    def dispatch(self, call):
        args = json.loads(call['function']['arguments'])
        name = call['function']['name']
        if name == 'query':
            return self.query(**args)
        if name == 'annotate':
            feedback = self.feedback(args.pop('version', None))
            qid = args.pop('qid')
            value = feedback.validate_annotation(qid, args)
            # Each revision is retained; the current pointer is for queries only.
            write_once(feedback.annotations / 'history' / (digest(value) + '.json'), value)
            atomic_json(feedback.annotations / (safe_id(qid) + '.json'), value)
            return dict(saved=True, requirements=len(value['requirements']), judgment=value['basis'])
        if name == 'diagnose':
            record = dict(self.feedback().diagnose(args), version=self.state['current'])
            ident = digest(record)[:24]
            write_once(self.root / 'diagnoses' / (ident + '.json'), record)
            return dict(diagnosis_id=ident, metrics=record['metrics'], basis=record['basis'])
        if name == 'submit':
            return self.submit(call['id'], args)
        raise ValueError('Unknown optimizer tool')

    def optimize(self):
        import httpx
        c, state = self.config, self.state
        credentials = read_json(c['plan']['credentials'])
        while not state['stopped']:
            # Finish persisted tool calls before making another paid request.
            assistant = next((m for m in reversed(state['messages']) if m['role'] == 'assistant'), {})
            answered = {m['tool_call_id'] for m in state['messages'] if m['role'] == 'tool'}
            for call in assistant.get('tool_calls') or []:
                if call['id'] in answered:
                    continue
                try:
                    result = self.dispatch(call)
                except BudgetStop as exc:
                    state.update(stopped=True, stop_reason=str(exc))
                    break
                except (ValueError, KeyError, StopIteration, TypeError, FileNotFoundError) as exc:
                    result = dict(error=str(exc), effect='No successful tool result; inspect the error and revise the request')
                state['messages'].append(dict(role='tool', tool_call_id=call['id'], content=json.dumps(result, ensure_ascii=False)))
                self.save()
            if (state['stopped'] or state['calls'] >= c['optimizer']['max_calls'] or len(state['completed']) >= c['max_candidates']
                    or state['stalls'] >= c['patience']):
                break
            try:
                self.admit()
            except BudgetStop as exc:
                state['stop_reason'] = str(exc)
                break
            state['calls'] += 1
            self.save()
            ident = str(state['calls'])
            log = self.root / 'optimizer' / 'http.jsonl'
            body = dict(model=c['optimizer']['model'], messages=state['messages'], tools=TOOLS,
                        temperature=c['optimizer']['temperature'], max_tokens=c['optimizer']['max_output_tokens'])
            append(log, dict(event='request', http_id=ident, stage='optimizer', request=body, timestamp=now()))
            # No automatic retries: a transport failure has unknown billing.
            with httpx.Client(timeout=900, trust_env=False) as client:
                response = client.post(credentials['base_url'].rstrip('/') + '/chat/completions',
                    headers={'Authorization': 'Bearer ' + credentials['api_key']}, json=body)
                payload = response.json()
            append(log, dict(event='response', http_id=ident, stage='optimizer', http_status=response.status_code,
                             response=payload, usage=payload.get('usage'), timestamp=now()))
            response.raise_for_status()
            if payload['model'] != c['optimizer']['response_model']:
                raise RuntimeError('Optimizer response model mismatch')
            choice = payload['choices'][0]
            if choice['finish_reason'] not in ('stop', 'tool_calls'):
                raise RuntimeError('Optimizer output incomplete; retained for inspection')
            message = choice['message']
            state['messages'].append(message)
            state['stopped'] = not bool(message.get('tool_calls'))
            self.save()
        state['stopped'] = True
        self.save()
        write_once(self.root / 'frozen.json', dict(version=state['current'],
            code=manifest(self.code(state['current'])), rsi_signature=c['signature']))
        self.budget()
        print('Frozen', state['current'], '(test set has not been run)')

    def test(self):
        frozen = read_json(self.root / 'frozen.json')
        version = frozen['version']
        if frozen['code'] != manifest(self.code(version)):
            raise ValueError('Frozen version changed')
        values = self.results('baseline', 'test') if version == 'baseline' else self.evaluate(version, 'test')
        result = comparison(self.results('baseline', 'test'), self.results('baseline', 'test'), values,
                            self.config['quality_tolerance'], self.config['min_saving'])
        write_once(self.root / 'test.json', dict(version=version, results=result,
            note='Held-out point estimates; formal grouped uncertainty and repeated RSI runs are still required'))
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('init')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--optimizer-model', default=MODEL)
    p.add_argument('--optimizer-response-model', default=RESPONSE_MODEL)
    p.add_argument('--max-candidates', type=int, default=3)
    p.add_argument('--max-calls', type=int, default=30)
    p.add_argument('--max-total-tokens', type=int, default=5000000)
    p.add_argument('--quality-tolerance', type=float, default=0)
    p.add_argument('--min-saving', type=float, default=0.05)
    for command in ('run', 'test', 'query'):
        p = commands.add_parser(command)
        p.add_argument('--root', type=Path, required=True)
        if command == 'query':
            p.add_argument('--kind', choices=['cases', 'case', 'event', 'source', 'evidence', 'cost', 'code', 'diagnoses'], required=True)
            p.add_argument('--qid')
            p.add_argument('--ref')
            p.add_argument('--version')
            p.add_argument('--offset', type=int, default=0)
            p.add_argument('--limit', type=int, default=6000)
        else:
            p.add_argument('--execute', action='store_true', help='Authorize real API spending')
    args = parser.parse_args()
    if args.command == 'init':
        if min(args.max_calls, args.max_candidates, args.max_total_tokens) < 1 or not 0 <= args.quality_tolerance <= 1 or not 0 < args.min_saving < 1:
            parser.error('Positive budgets, quality tolerance in [0,1], and saving in (0,1) required')
        initialize(args)
        return
    controller = Controller(args.root)
    if args.command == 'query':
        print(json.dumps(controller.query(args.kind, args.qid, args.ref, args.offset, args.limit, args.version), ensure_ascii=False, indent=2))
        return
    if not args.execute:
        parser.error('run/test spend real API money; pass --execute explicitly')
    if sys.platform != 'linux' or not shutil.which('bwrap') or not (ROOT / '.venv/bin/python').exists():
        parser.error('Execution requires Linux, bubblewrap and the repository .venv')
    # Verify original bytes before mounting data. No LLM calls here.
    from experiment import validate_input, configuration
    if digest(configuration(controller.config['plan'])) != controller.config['baseline_signature']:
        raise ValueError('Installed packages or runtime configuration differ from the baseline')
    validate_input(controller.config['source'], controller.config['plan']['datasets'][0]['expected_qa_count'])
    if sha(controller.config['parquet']) != controller.config['parquet_sha256']:
        raise ValueError('Baseline corpus parquet changed')
    if args.command == 'run':
        controller.optimize()
    else:
        controller.test()


if __name__ == '__main__':
    main()
