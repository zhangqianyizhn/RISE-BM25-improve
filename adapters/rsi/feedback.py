"""Evidence-state and cost calculations over original RISE records (no LLM calls)."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiment'))
from common import digest, read_json, safe_id


def sha(path):
    # Corpora, Parquet and HTTP histories can each be multiple GB.
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def events(path):
    if Path(path).exists():
        with Path(path).open(encoding='utf-8') as stream:
            for line, text in enumerate(stream, 1):
                if text.strip():
                    yield line, json.loads(text)


def checked_results(dataset, ids, signature):
    """Never drop failures, stale answers, missing usage, or unscored questions."""
    result = {}
    for qid in ids:
        root = Path(dataset) / 'qa' / safe_id(qid)
        r, e = read_json(root / 'result.json'), read_json(root / 'evaluation.json')
        if (r['id'] != qid or r['status'] != 'complete' or e['status'] != 'complete'
                or r['signature'] != signature or e['signature'] != signature
                or e['answer_hash'] != digest(r['answer']) or not r['answer'].strip()
                or not r['usage']['usage_complete'] or r.get('protocol_errors')
                or type(e['score']) is not int or not 0 <= e['score'] <= 4):
            raise ValueError('Incomplete or stale evaluation: ' + qid)
        result[qid] = dict(score=e['score'], tokens=r['usage']['total_tokens'],
                           seconds=r['elapsed_seconds'])
    return result


def comparison(initial, current, candidate, quality_tolerance, min_saving):
    if not initial or initial.keys() != current.keys() or current.keys() != candidate.keys():
        raise ValueError('Paired evaluations must contain exactly the same nonempty question set')
    n = len(initial)
    quality = Fraction(sum(candidate[k]['score'] - initial[k]['score'] for k in initial), 4 * n)
    old_tokens = sum(x['tokens'] for x in current.values())
    new_tokens = sum(x['tokens'] for x in candidate.values())
    saving = Fraction(old_tokens - new_tokens, old_tokens) if old_tokens else Fraction(0)
    return dict(accepted=quality >= -Fraction(str(quality_tolerance)) and saving >= Fraction(str(min_saving)),
                quality_change_from_initial=float(quality), token_saving_from_current=float(saving),
                total_tokens_saved_from_current=old_tokens - new_tokens,
                mean_tokens=new_tokens / n,
                mean_score=sum(x['score'] for x in candidate.values()) / n,
                regressed_questions=sum(candidate[k]['score'] < initial[k]['score'] for k in initial),
                n=n, inference='pilot point estimates, not a non-inferiority significance claim')


def audit_usage(root):
    """All HTTP sends, including failed attempts and preflight/judge requests."""
    totals = dict(input_tokens=0, output_tokens=0, total_tokens=0, requests=0,
                  unknown_requests=0, by_stage={})
    for path in Path(root).rglob('http.jsonl'):
        pending = {}
        seen = set()
        for _, event in events(path):
            ident = event['http_id']
            if event['event'] == 'request':
                if ident in seen:
                    raise ValueError('Duplicate HTTP request ID: ' + ident)
                seen.add(ident)
                pending[ident] = event.get('stage', 'unknown')
                totals['requests'] += 1
            elif event['event'] == 'response':
                u = event.get('usage') or {}
                values = [u.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
                if all(type(v) is int and v >= 0 for v in values) and values[0] + values[1] == values[2]:
                    if ident not in pending:
                        raise ValueError('Unmatched or duplicate HTTP response: ' + ident)
                    stage = pending.pop(ident)
                    if 'preflight' in path.relative_to(root).parts:
                        stage = 'preflight_' + stage
                    for name, value in zip(('input_tokens', 'output_tokens', 'total_tokens'), values):
                        totals[name] += value
                    totals['by_stage'][stage] = totals['by_stage'].get(stage, 0) + values[2]
        totals['unknown_requests'] += len(pending)
    totals['usage_complete'] = totals['unknown_requests'] == 0
    return totals


class Feedback:
    def __init__(self, dataset, source, questions, signature, scoring=None):
        self.dataset, self.source = Path(dataset), Path(source)
        self.scoring = Path(scoring) if scoring else self.dataset
        self.questions = {q['id']: q for q in questions}
        self.signature = signature
        self.documents = {d['id']: d for _, d in events(self.source / 'documents.jsonl')}

    def attempt(self, qid):
        if qid not in self.questions:
            raise ValueError('Only development questions are queryable')
        result = read_json(self.dataset / 'qa' / safe_id(qid) / 'result.json')
        if result['signature'] != self.signature:
            raise ValueError('Trace signature mismatch')
        path = (self.dataset / result['attempt']).resolve()
        if not path.is_relative_to(self.dataset.resolve()):
            raise ValueError('Attempt outside dataset')
        return path

    def overview(self):
        values = checked_results(self.scoring, self.questions, self.signature)
        return [dict(id=k, **v) for k, v in sorted(values.items(), key=lambda x: -x[1]['tokens'])]

    def evidence(self, qid, annotation=None):
        attempt = self.attempt(qid)
        # Optional analysis input, not an annotation/diagnosis admission gate.
        # Without quotes, track known sources only; S4 and S5 stay unassessed.
        annotation = json.loads(json.dumps(annotation or {}))
        if 'requirements' not in annotation:
            annotation['requirements'] = [dict(fact='Known source: ' + doc_id,
                alternatives=[[dict(doc_id=doc_id, quote='')]])
                for doc_id in self.questions[qid].get('document_ids', [])]
        for requirement in annotation['requirements']:
            for group in requirement['alternatives']:
                for span in group:
                    span.setdefault('quote', '')
                    span['span_id'] = digest([span['doc_id'], span['quote']])[:24]
        annotation['basis'] = 'Analysis supplied by optimizer; source quotes and coverage are not certified'
        # Workspace snapshots have call IDs, but old logs have no request IDs.
        # Join them to tool returns, then bound by actual request timestamps.
        returns = {e['call_id']: e['timestamp'] for _, e in events(attempt / 'tools.jsonl')
                   if e['event'] == 'return'}
        workspaces = [(returns[e['call_id']], set(e['paths']))
                      for _, e in events(attempt / 'workspace.jsonl') if e['call_id'] in returns]
        judgments = {j['http_id']: j for j in annotation.get('sufficiency', [])}
        observations = []
        for line, event in events(attempt / 'http.jsonl'):
            if event['event'] != 'request':
                continue
            messages = event['request']['messages']
            visible = [m.get('content') or '' for m in messages if m['role'] == 'tool']
            reachable = set()
            for stamp, paths in workspaces:
                if stamp <= event['timestamp']:
                    reachable = paths
            states = []
            for requirement in annotation['requirements']:
                alternatives = []
                for group in requirement['alternatives']:
                    branch = []
                    for span in group:
                        rel = str(Path(self.documents[span['doc_id']]['path']).relative_to('corpus'))
                        state = ('S4' if span['quote'] and any(span['quote'] in t for t in visible) else
                                 'S3' if any(rel in t for t in visible) else
                                 'S2' if rel in reachable else
                                 'S1' if (attempt / 'workspace.jsonl').exists() else 'unknown')
                        branch.append(dict(span_id=span['span_id'], state=state,
                                           quote_supplied=bool(span['quote'])))
                    alternatives.append(branch)
                states.append(dict(fact=requirement['fact'], alternatives=alternatives,
                                   visible=any(g and all(s['state'] == 'S4' for s in g) for g in alternatives)))
            observations.append(dict(http_id=event['http_id'], ref=f'http.jsonl:{line}', requirements=states,
                sufficiency=dict(judgments.get(event['http_id'], {'verdict': 'unknown'}),
                                 basis='optimizer judgment, not a deterministic state'),
                all_annotated_requirements_visible=bool(states) and all(s['visible'] for s in states)))
        result = read_json(attempt / 'result.json')
        return dict(annotation=annotation, observations=observations,
                    terminal=dict(state='S6', status=result.get('status'), answer=result.get('answer')),
                    limitations=['S1 concerns known sources only', 'S3 is a literal path mention',
                                 'S4 checks supplied quote visibility, not its source validity, adoption or sufficiency',
                                 'Without supplied quotes S4 is unassessed, not evidence of missing facts',
                                 'States are observed lower bounds; absent exact matches do not prove absent evidence',
                                 'S5 and source adoption remain unknown without semantic review'])

    def cost(self, qid, evidence=None):
        attempt = self.attempt(qid)
        try:
            import tiktoken
            encoding = tiktoken.get_encoding('cl100k_base')
        except (ImportError, OSError):
            encoding = None
        evidence = evidence if evidence is not None else self.evidence(qid)
        observations = {o['http_id']: o for o in evidence['observations']}
        spans = {s['span_id']: s['quote'] for r in evidence['annotation']['requirements']
                 for group in r['alternatives'] for s in group if s['quote']}
        responses = {e['http_id']: e.get('usage') for _, e in events(attempt / 'http.jsonl') if e['event'] == 'response'}
        residence = [dict(fact=r['fact'], states={}) for r in evidence['annotation']['requirements']]
        items, requests, previously_visible = {}, [], set()
        for line, event in events(attempt / 'http.jsonl'):
            if event['event'] != 'request':
                continue
            messages = event['request']['messages']
            operations = {t['id']: t['function'] for m in messages for t in m.get('tool_calls') or []}
            duplicates = Counter()
            carried, content_ids = 0, []
            for index, message in enumerate(messages):
                if message['role'] != 'tool':
                    continue
                text = message.get('content') or ''
                key = (message.get('tool_call_id'), digest(text))
                duplicates[key] += 1
                occurrence = digest([*key, duplicates[key]])[:24]
                if occurrence not in items:
                    items[occurrence] = dict(occurrence_id=occurrence, tool_call_id=key[0],
                        content_hash=key[1], first_ref=f'http.jsonl:{line}', message_index=index,
                        chars=len(text), tokens_estimate=len(encoding.encode(text, disallowed_special=())) if encoding else None,
                        operation=operations.get(key[0]), matched_span_ids=[s for s, quote in spans.items() if quote in text],
                        http_ids=[], requests=0)
                items[occurrence]['requests'] += 1
                items[occurrence]['http_ids'].append(event['http_id'])
                items[occurrence]['last_ref'] = f'http.jsonl:{line}'
                content_ids.append(occurrence)
                carried += items[occurrence]['tokens_estimate'] or 0
            usage = responses.get(event['http_id']) or {}
            values = [usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
            if not (all(type(v) is int and v >= 0 for v in values) and values[0] + values[1] == values[2]):
                usage = None
            observation = observations[event['http_id']]
            visible = {i for i, r in enumerate(observation['requirements']) if r['visible']}
            for i, requirement in enumerate(observation['requirements']):
                # AND spans use their lowest observed boundary; OR alternatives
                # use their highest. Each requirement has its own overlapping bill.
                rank = max((min((int(s['state'][1:]) if s['state'] != 'unknown' else 0 for s in group), default=0)
                            for group in requirement['alternatives']), default=0)
                bucket = residence[i]['states'].setdefault('S' + str(rank) if rank else 'unknown',
                    dict(requests=0, known_total_tokens=0, unknown_usage_requests=0))
                bucket['requests'] += 1
                bucket['known_total_tokens'] += usage['total_tokens'] if usage else 0
                bucket['unknown_usage_requests'] += int(usage is None)
            requests.append(dict(http_id=event['http_id'], ref=f'http.jsonl:{line}',
                usage=usage, content_ids=content_ids,
                carried_tool_tokens_estimate=carried if encoding else None,
                literal_visible_requirements=sorted(visible), newly_visible_requirements=sorted(visible - previously_visible),
                no_longer_visible_requirements=sorted(previously_visible - visible), sufficiency=observation['sufficiency']))
            previously_visible = visible
        for item in items.values():
            item['carried_chars'] = item['chars'] * item['requests']
            item['carried_tokens_estimate'] = (item['tokens_estimate'] * item['requests']
                                               if encoding else None)
        return dict(actual_accepted_qa=read_json(attempt / 'result.json')['usage'],
                    all_attempts=audit_usage(attempt.parent), requests=requests, requirement_residence=residence,
                    content=sorted(items.values(), key=lambda x: -x['carried_chars']),
                    estimate_tokenizer='cl100k_base (not provider billing)' if encoding else None,
                    limitations=['Tool content only; assistant/system/envelope input remains unattributed',
                                 'Physical duplicate messages count separately; retries count as sends',
                                 'Residence bills overlap across requirements; never sum them across facts',
                                 'First/last refs locate input visibility, not tool execution; operations may be unknown',
                                 'No literal progress or matched span does not establish usefulness or necessity',
                                 'Carrying burden is not removable cost; no dollar estimate'])

    def metrics(self, qid, annotation=None):
        evidence = self.evidence(qid, annotation)
        cost = self.cost(qid, evidence)
        counts = Counter()
        for requirement in cost['requirement_residence']:
            for state, value in requirement['states'].items():
                counts[state] += value['requests']
        sufficiency = Counter(o['sufficiency']['verdict'] for o in evidence['observations'])
        quoted = sum(any(g and all(s['quote'] for s in g) for g in r['alternatives'])
                     for r in evidence['annotation']['requirements'])
        summary = dict(qid=qid, actual_accepted_qa=cost['actual_accepted_qa'],
            all_attempts=cost['all_attempts'],
            states={**{s: dict(requirement_requests=counts[s]) for s in ('S1', 'S2', 'S3', 'S4')},
                    'S5': dict(request_judgments=dict(sufficiency), basis='optimizer judgment'),
                    'S6': dict(status=evidence['terminal']['status'])},
            requirements=len(evidence['annotation']['requirements']), quoted_requirements=quoted,
            unknown_requirement_requests=counts['unknown'],
            carried_tool_chars=sum(c['carried_chars'] for c in cost['content']),
            note='S1–S4 count overlapping requirement/request observations, not disjoint turns or savings; '
                 'S4 is unassessed for requirements without quotes; S5 defaults to unknown')
        return dict(summary=summary, evidence=evidence, cost=cost)
