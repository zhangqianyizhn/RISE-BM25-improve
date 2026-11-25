"""Read-only, paginated feedback over existing RISE records (no API calls)."""
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


def page(value, offset=0, limit=6000):
    """Offsets count Unicode characters, including for serialized JSON."""
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 12000:
        raise ValueError('Use offset >= 0 and limit between 1 and 12000')
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    end = min(len(text), offset + limit)
    return dict(text=text[offset:end], offset=offset, total_chars=len(text),
                next_offset=end if end < len(text) else None, sha256=digest(text))


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
    def __init__(self, dataset, source, questions, signature, annotations, scoring=None):
        self.dataset, self.source = Path(dataset), Path(source)
        self.scoring = Path(scoring) if scoring else self.dataset
        self.questions = {q['id']: q for q in questions}
        self.signature, self.annotations = signature, Path(annotations)
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

    def document(self, doc_id):
        d = self.documents[doc_id]
        path = (self.source / d['path']).resolve()
        if not path.is_relative_to((self.source / 'corpus').resolve()) or sha(path) != d['sha256']:
            raise ValueError('Source changed or escaped corpus')
        return path.read_bytes().decode('utf-8'), d

    def overview(self):
        values = checked_results(self.scoring, self.questions, self.signature)
        return [dict(id=k, **v) for k, v in sorted(values.items(), key=lambda x: -x[1]['tokens'])]

    def annotation(self, qid):
        path = self.annotations / (safe_id(qid) + '.json')
        return read_json(path) if path.exists() else dict(requirements=[], coverage_complete=False)

    def validate_annotation(self, qid, value):
        attempt = self.attempt(qid)
        requirements = value['requirements']
        if not requirements or len(requirements) > 30 or type(value['coverage_complete']) is not bool:
            raise ValueError('Supply 1–30 requirements and a boolean coverage_complete judgment')
        for requirement in requirements:
            if not requirement['fact'].strip() or not requirement['alternatives']:
                raise ValueError('Each requirement needs a fact and alternative support groups')
            for group in requirement['alternatives']:
                if not group:
                    raise ValueError('An alternative must contain supporting spans')
                for span in group:
                    text, d = self.document(span['doc_id'])
                    quote = span['quote']
                    if not isinstance(quote, str) or not quote or text.count(quote) != 1:
                        raise ValueError('Quote must occur exactly once; include enough surrounding text')
                    start = text.index(quote)
                    span.update(source_hash=d['sha256'], start=start, end=start + len(quote),
                                span_id=digest([d['sha256'], start, quote])[:24])
        if value.get('sufficiency'):
            requests = {e['http_id'] for _, e in events(attempt / 'http.jsonl') if e['event'] == 'request'}
            for judgment in value['sufficiency']:
                if (judgment['http_id'] not in requests or not judgment['reason'].strip()
                        or judgment['verdict'] not in {'sufficient', 'insufficient', 'unknown'}):
                    raise ValueError('Sufficiency needs an actual request ID, verdict and reason')
        return dict(**value, case_id=qid, trace_signature=self.signature,
                    basis='optimizer semantic judgment; spans verified mechanically; not ground truth')

    def evidence(self, qid):
        attempt = self.attempt(qid)
        annotation = self.annotation(qid)
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
                        state = ('S4' if any(span['quote'] in t for t in visible) else
                                 'S3' if any(rel in t for t in visible) else
                                 'S2' if rel in reachable else
                                 'S1' if (attempt / 'workspace.jsonl').exists() else 'unknown')
                        branch.append(dict(span_id=span['span_id'], state=state))
                    alternatives.append(branch)
                states.append(dict(fact=requirement['fact'], alternatives=alternatives,
                                   visible=any(all(s['state'] == 'S4' for s in g) for g in alternatives)))
            observations.append(dict(http_id=event['http_id'], ref=f'http.jsonl:{line}', requirements=states,
                sufficiency=dict(judgments.get(event['http_id'], {'verdict': 'unknown'}),
                                 basis='optimizer judgment, not a deterministic state'),
                all_annotated_requirements_visible=bool(states) and all(s['visible'] for s in states)))
        return dict(annotation=annotation, observations=observations,
                    terminal=dict(state='S6', answer=read_json(attempt / 'result.json').get('answer')),
                    limitations=['S1 concerns known sources only', 'S3 is a literal path mention',
                                 'S4 verifies quote visibility, not adoption or full semantic sufficiency',
                                 'States are observed lower bounds; absent exact matches do not prove absent evidence',
                                 'S5 and source adoption remain unknown without semantic review'])

    def cost(self, qid):
        attempt = self.attempt(qid)
        try:
            import tiktoken
            encoding = tiktoken.get_encoding('cl100k_base')
        except ImportError:
            encoding = None
        evidence = self.evidence(qid)
        observations = {o['http_id']: o for o in evidence['observations']}
        spans = {s['span_id']: s['quote'] for r in evidence['annotation']['requirements']
                 for group in r['alternatives'] for s in group}
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
                rank = max(min(int(s['state'][1:]) if s['state'] != 'unknown' else 0 for s in group)
                           for group in requirement['alternatives'])
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

    def diagnose(self, value):
        """Ground a mechanism and its cost scope; never certify its causal claim."""
        for name in ('observations', 'hypothesis', 'counterevidence', 'unknowns', 'expected_change', 'risks'):
            if not value[name].strip():
                raise ValueError('Diagnosis needs ' + name + '; explicitly state unknown when unresolved')
        cases = value['cases']
        if not cases or len({c['qid'] for c in cases}) != len(cases):
            raise ValueError('Diagnosis needs distinct development cases')
        grounded, tokens, unknown, burden = [], 0, 0, 0
        for case in cases:
            annotation = self.annotation(case['qid'])
            if not annotation['requirements']:
                raise ValueError('Annotate cited development evidence before diagnosis')
            if not case['evidence_refs'] or not case['cost_http_ids']:
                raise ValueError('Cite original event lines and request IDs from the cost query')
            refs = []
            for ref in case['evidence_refs']:
                if ':' not in ref:
                    raise ValueError('Evidence reference must name a trace line')
                event = self.query('event', qid=case['qid'], ref=ref)
                refs.append(dict(ref=ref, event_hash=event['sha256']))
            if (case['necessity'] not in {'necessary', 'avoidable', 'mixed', 'unknown'}
                    or not case['reason'].strip()):
                raise ValueError('Cost necessity needs a judgment and its evidence-based reason')
            cost = self.cost(case['qid'])
            requests = {r['http_id']: r for r in cost['requests']}
            content = {c['occurrence_id']: c for c in cost['content']}
            ids, content_ids = set(case['cost_http_ids']), set(case['content_ids'])
            if not ids <= requests.keys() or not content_ids <= content.keys():
                raise ValueError('Use actual request and content IDs from this case cost query')
            if any(not ids.intersection(content[k]['http_ids']) for k in content_ids):
                raise ValueError('Selected content must occur in the selected requests')
            selected = [requests[k] for k in sorted(ids)]
            amount = sum(r['usage']['total_tokens'] for r in selected if r['usage'])
            missing = sum(r['usage'] is None for r in selected)
            carried = sum(content[k]['chars'] * len(ids.intersection(content[k]['http_ids'])) for k in content_ids)
            matched = sorted({span for k in content_ids for span in content[k]['matched_span_ids']})
            grounded.append(dict(**case, verified_refs=refs, annotation_hash=digest(annotation),
                cost_scope=dict(known_associated_tokens=amount, unknown_usage_requests=missing,
                                selected_carried_chars=carried, matched_span_ids=matched),
                requests=selected, annotation=annotation))
            tokens, unknown, burden = tokens + amount, unknown + missing, burden + carried
        return dict(claim=value, cases=grounded, trace_signature=self.signature,
            metrics=dict(affected_cases=len(cases), known_associated_tokens=tokens,
                         unknown_usage_requests=unknown, selected_carried_chars=burden),
            basis='References and cost scope verified; mechanism and necessity are optimizer judgments. '
                  'Associated tokens and carrying are not savings; compare paired QA results to test the claim.')

    def query(self, kind, qid=None, ref=None, offset=0, limit=6000):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 12000:
            raise ValueError('Use offset >= 0 and limit between 1 and 12000')
        if kind == 'cases':
            value = self.overview()
        elif kind == 'source':
            value, _ = self.document(ref)
        elif kind == 'case':
            self.attempt(qid)
            value = self.questions[qid]
        elif kind == 'evidence':
            value = self.evidence(qid)
            if ref:
                value = next((o for o in value['observations'] if o['http_id'] == ref), None)
                if value is None:
                    raise ValueError('Unknown evidence request ID')
        elif kind == 'cost':
            value = self.cost(qid)
            if ref:
                value = next((v for v in value['requests'] + value['content']
                              if v.get('http_id', v.get('occurrence_id')) == ref), None)
                if value is None:
                    raise ValueError('Unknown cost request or content ID')
        elif kind == 'event':
            name, _, number = ref.partition(':')
            if name not in {'http.jsonl', 'tools.jsonl', 'retrieval.jsonl', 'workspace.jsonl', 'native_full.json'}:
                raise ValueError('Unknown trace file')
            path = self.attempt(qid) / name
            if number:
                value = next(e for line, e in events(path) if line == int(number))
            elif name.endswith('.json'):
                value = read_json(path)
            else:
                # Preserve original text and supply actual line references, without
                # building the whole trace or asking the optimizer to count lines.
                chunk, skip = '', offset
                with path.open(encoding='utf-8') as stream:
                    for line, text in enumerate(stream, 1):
                        text = f'{name}:{line} ' + text
                        if skip >= len(text):
                            skip -= len(text)
                            continue
                        chunk += text[skip:skip + limit + 1 - len(chunk)]
                        skip = 0
                        if len(chunk) > limit:
                            break
                return dict(text=chunk[:limit], offset=offset,
                            next_offset=offset + limit if len(chunk) > limit else None,
                            ref=ref, trace_signature=self.signature)
        else:
            raise ValueError('Unknown query kind')
        return dict(page(value, offset, limit), ref=ref, trace_signature=self.signature)
