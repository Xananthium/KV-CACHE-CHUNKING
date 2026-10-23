#!/usr/bin/env python3
"""Generate work and a durable summary, then compare cached vs fresh summaries.

CPU research only. The summary cache is retained from actual generation; no
prefill of the discarded work is used to build it again. Tensor concatenation
here is an experimental control, not the zero-copy production bank interface.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import platform

import torch
import transformers

from run_experiment import (MODEL_ID, REVISION, Runner, cache_bytes, cat_cache,
                            ids, prefix, rebase_cache, slice_cache)


def chat(role, text):
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def generate(runner, cache, logits, limit):
    tokens = []
    stopped = False
    start = time.perf_counter()
    for _ in range(limit):
        token = int(logits.argmax())
        tokens.append(token)
        # Include the final generated token in the retained cache, including EOS.
        cache, logits = runner.forward([token], cache)
        if token in runner.eos:
            stopped = True
            break
    return tokens, cache, logits, stopped, (time.perf_counter() - start) * 1000


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--max-new-tokens', type=int, default=12)
    parser.add_argument('--work-tokens', type=int, default=256)
    parser.add_argument('--summary-tokens', type=int, default=160)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'rows.jsonl').exists():
        raise SystemExit('Choose a fresh output directory; existing results are preserved.')
    runner = Runner(args)
    rng = random.Random(args.seed)
    p = ids(runner.tokenizer, chat('system', 'You are a precise ledger assistant. '
        'Use the supplied account records. Follow each request exactly.'))
    shared_cache, _ = runner.forward(p)
    rows, cases = [], []
    started = time.perf_counter()
    for case_id in range(args.count):
        opening, deposit, withdrawal = rng.randint(100, 400), rng.randint(10, 90), rng.randint(10, 90)
        closing = opening + deposit - withdrawal
        ledger = (f'The Cedar account opens with {opening} credits, receives a deposit '
                  f'of {deposit} credits, and pays a withdrawal of {withdrawal} credits. '
                  'Work through the arithmetic and verify the closing balance.')
        work_prompt = ids(runner.tokenizer, chat('user', ledger) + '<|im_start|>assistant\n')
        stage = time.perf_counter()
        work_cache, work_logits = runner.forward(work_prompt, shared_cache)
        work, work_cache, _, work_stopped, work_decode_ms = generate(runner, work_cache, work_logits, args.work_tokens)
        work_total_ms = (time.perf_counter() - stage) * 1000
        # Explicitly close a truncated work response; record that truncation.
        work_suffix = '\n' if work_stopped else '<|im_end|>\n'
        for summary_kind in ('all_facts', 'closing_only'):
            instruction = ('Compact the preceding record and work into one short durable note. '
                           'Preserve account name, opening balance, deposit, withdrawal, and closing balance. '
                           'State the numbers; omit calculations and commentary.')
            if summary_kind == 'closing_only':
                instruction = ('Compact the preceding work into one short durable note containing only '
                               'the account name and closing balance. Omit the opening balance, transactions, '
                               'calculations, and commentary.')
            bridge = ids(runner.tokenizer, work_suffix + chat('user', instruction))
            c_header = ids(runner.tokenizer, '<|im_start|>assistant\n')
            r = work_prompt + work + bridge
            stage = time.perf_counter()
            before_c, _ = runner.forward(bridge, work_cache)
            c_cache, c_logits = runner.forward(c_header, before_c)
            summary, retained_full, _, summary_stopped, summary_decode_ms = generate(
                runner, c_cache, c_logits, args.summary_tokens)
            summary_total_ms = (time.perf_counter() - stage) * 1000
            c = c_header + summary
            # Complete the chat record without rebuilding any generated KV.
            ending = ids(runner.tokenizer, '\n' if summary_stopped else '<|im_end|>\n')
            retained_full, _ = runner.forward(ending, retained_full)
            c += ending
            assert retained_full[0][0].shape[2] == len(p) + len(r) + len(c)
            conclusion_cache = slice_cache(retained_full, len(p) + len(r), len(p) + len(r) + len(c))
            text = runner.tokenizer.decode(summary, skip_special_tokens=True)
            case = {'case_id': case_id, 'kind': summary_kind, 'opening': opening,
                    'deposit': deposit, 'withdrawal': withdrawal, 'closing': closing,
                    'work': runner.tokenizer.decode(work, skip_special_tokens=True),
                    'summary': text, 'work_stopped_on_eos': work_stopped,
                    'summary_stopped_on_eos': summary_stopped,
                    'work_total_ms': work_total_ms, 'work_decode_ms': work_decode_ms,
                    'summary_total_ms': summary_total_ms, 'summary_decode_ms': summary_decode_ms,
                    'tokens': {'p': p, 'r': r, 'c': c},
                    'summary_contains_opening_literal': str(opening) in text,
                    'summary_contains_closing_literal': str(closing) in text}
            cases.append(case)
            for question_kind, gold in [('closing', str(closing)), ('opening', str(opening))]:
                q = ids(runner.tokenizer, chat('user', f'What was the {question_kind} balance of the Cedar account? '
                                             'Reply with the number only.') + '<|im_start|>assistant\n')
                for method in ('full_generated_branch', 'fresh_generated_summary', 'retained_summary_holes', 'retained_summary_rebased'):
                    stage = time.perf_counter()
                    next_position = None
                    if method == 'full_generated_branch':
                        keep = retained_full
                        cache, logits = runner.forward(q, keep)
                    elif method == 'fresh_generated_summary':
                        # Common P is already cached in both systems. Recompute C only.
                        keep, _ = runner.forward(c, shared_cache)
                        cache, logits = runner.forward(q, keep)
                    elif method == 'retained_summary_holes':
                        keep = cat_cache(shared_cache, conclusion_cache)
                        absolute = len(p) + len(r) + len(c)
                        cache, logits = runner.forward(q, keep, position_start=absolute)
                        next_position = absolute + len(q)
                    else:
                        keep = cat_cache(shared_cache, rebase_cache(conclusion_cache, -len(r), runner.inv_freq))
                        cache, logits = runner.forward(q, keep)
                    online_ms = (time.perf_counter() - stage) * 1000
                    scored = runner.score_and_decode(cache, logits, gold, next_position)
                    row = {'case_id': case_id, 'summary_kind': summary_kind,
                           'question_kind': question_kind, 'method': method, 'gold': gold,
                           'work_tokens': len(r), 'summary_tokens': len(c),
                           'logical_retained_cache_bytes': cache_bytes(keep),
                           'online_assembly_and_prefill_ms': online_ms,
                           'work_build_ms': work_total_ms, 'summary_build_ms': summary_total_ms,
                           'summary_stopped_on_eos': summary_stopped, **scored}
                    rows.append(row)
                    with (args.output / 'rows.jsonl').open('a') as log:
                        log.write(json.dumps(row) + '\n')
            (args.output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
            print(json.dumps({'case_id': case_id, 'kind': summary_kind,
                              'summary_tokens': len(c), 'summary_stopped': summary_stopped}), flush=True)
    summaries = []
    for kind in ('all_facts', 'closing_only'):
        for question in ('opening', 'closing'):
            for method in sorted(set(r['method'] for r in rows)):
                group = [r for r in rows if (r['summary_kind'], r['question_kind'], r['method']) == (kind, question, method)]
                summaries.append({'summary_kind': kind, 'question_kind': question, 'method': method,
                                  'n': len(group), 'exact_answers': sum(r['exact_answer'] for r in group)})
    metadata = {'model': MODEL_ID, 'revision': REVISION, 'device': 'cpu', 'dtype': 'float32',
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'python': platform.python_version(),
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'elapsed_seconds': time.perf_counter() - started, 'rows': len(rows),
                'limits': ['Small greedy CPU model, not production Gemma or vLLM.',
                           'Generated summaries can contain errors; all cases retained, including truncations.',
                           'Logical retained KV bytes do not establish physical allocation savings in this concatenating harness.',
                           'Summary/work generation costs recorded separately; no end-to-end speedup assumed.',
                           'No discarded work was prefilled again to obtain the retained generated summary KV.']}
    for name, value in [('summary.json', summaries), ('metadata.json', metadata)]:
        (args.output / name).write_text(json.dumps(value, indent=2) + '\n')
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
