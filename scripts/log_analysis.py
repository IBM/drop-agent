#!/usr/bin/env python3
"""
Analyze BCP evaluation logs across multiple systems.

Usage:
    python scripts/log_analysis.py --folders results/system_a results/system_b
    python scripts/log_analysis.py --folders results/system_a results/system_b --sample failed
"""

import argparse
import json
import os
import glob
import random
import re
import statistics as _statistics


def ends_with_assistant(rec):
    messages = rec.get('messages', [])
    last_msg = messages[-1] if messages else None
    if not last_msg:
        return False
    return last_msg.get('role') == 'assistant' and last_msg.get('channel') == 'final'


def has_exact_answer(rec):
    messages = rec.get('messages', [])
    last_msg = messages[-1] if messages else None
    if not last_msg or last_msg.get('role') != 'assistant':
        return False
    content = last_msg.get('content', [])
    text = content[0].get('text', '') if content else ''
    return 'exact answer:' in text.lower()


def is_exact_match(rec):
    """Check if the ground-truth answer is contained in the model's Exact Answer field."""
    messages = rec.get('messages', [])
    last_msg = messages[-1] if messages else None
    if not last_msg or last_msg.get('role') != 'assistant':
        return False
    content = last_msg.get('content', [])
    text = content[0].get('text', '') if content else ''

    lower_text = text.lower()
    idx = lower_text.find('exact answer:')
    if idx == -1:
        return False
    answer_section = text[idx + len('exact answer:'):]
    next_section = answer_section.lower().find('\nconfidence:')
    if next_section != -1:
        answer_section = answer_section[:next_section]
    answer_section = answer_section.strip()

    ground_truth = rec.get('answer', '')
    if not ground_truth:
        return False
    return ground_truth.lower() in answer_section.lower()


def categorize_record(rec):
    """Return 'parses', 'completes', or 'failed' for sampling purposes."""
    if rec.get('status', 'success') != 'success':
        return 'failed'
    if has_exact_answer(rec):
        return 'parses'
    if ends_with_assistant(rec):
        return 'completes'
    return None


def load_accuracy(folder_path):
    """Extract Overall Accuracy percentage from results.txt, or None if not found."""
    results_path = os.path.join(folder_path, 'results.txt')
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        for line in f:
            if 'Overall Accuracy' in line:
                # Extract percentage like "50.24%"
                parts = line.split('%')
                if parts:
                    try:
                        pct = parts[0].split()[-1]
                        return float(pct)
                    except (ValueError, IndexError):
                        pass
    return None


def load_records(folder_path):
    pattern = os.path.join(folder_path, "node*.jsonl")
    seen_qids = {}
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    qid = rec.get('qid')
                    if qid in seen_qids:
                        raise ValueError(
                            f"Duplicate qid {qid!r} found in {path} "
                            f"(first seen in {seen_qids[qid]})"
                        )
                    seen_qids[qid] = path
                    records.append(rec)
    return records


def analyze_records(records):
    completes = 0   # ends with assistant message
    parses = 0      # ends with assistant message that has "Exact Answer:"
    em = 0          # answer field contained in model's Exact Answer section
    failed = 0      # status != success
    total_tool_calls = 0
    successful_tool_calls = 0
    char_counts = []

    for rec in records:
        if rec.get('status', 'success') != 'success':
            failed += 1
        if ends_with_assistant(rec):
            completes += 1
        if has_exact_answer(rec):
            parses += 1
        if is_exact_match(rec):
            em += 1

        messages = rec.get('messages', [])

        total_chars = 0
        for msg in messages:
            for item in msg.get('content', []):
                if isinstance(item, dict):
                    total_chars += len(item.get('text', ''))
                else:
                    total_chars += len(str(item))
        char_counts.append(total_chars)

        for msg in messages:
            if msg.get('role') == 'tool':
                total_tool_calls += 1
                has_error = False
                for item in msg.get('content', []):
                    text = item.get('text', '') if isinstance(item, dict) else str(item)
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict) and 'error' in parsed:
                            has_error = True
                            break
                    except (json.JSONDecodeError, ValueError):
                        pass
                if not has_error:
                    successful_tool_calls += 1

    avg_chars = int(sum(char_counts) / len(char_counts)) if char_counts else 0
    max_chars = max(char_counts) if char_counts else 0

    return {
        'total_questions': len(records),
        'completes': completes,
        'parses': parses,
        'em': em,
        'failed': failed,
        'total_tool_calls': total_tool_calls,
        'successful_tool_calls': successful_tool_calls,
        'avg_chars': avg_chars,
        'max_chars': max_chars,
    }


def _group_rows(rows):
    """Group rows where multiple names share a base differing only by trailing -\\d+."""
    potential = {}
    for name, _s, _a in rows:
        m = re.match(r'^(.*)-\d+$', name)
        if m:
            potential.setdefault(m.group(1), []).append(name)

    seen = {}
    order = []
    for name, stats, acc in rows:
        m = re.match(r'^(.*)-\d+$', name)
        key = m.group(1) if m and len(potential.get(m.group(1), [])) > 1 else name
        if key not in seen:
            seen[key] = []
            order.append(key)
        seen[key].append((name, stats, acc))

    return [(key, seen[key]) for key in order]


def _avg_stats(items):
    """Compute averaged stats and acc mean/std from a list of (name, stats, acc) items."""
    stats_list = [s for _, s, _ in items]
    accs = [a for _, _, a in items if a is not None]

    def avg(field):
        return sum(s[field] for s in stats_list) / len(stats_list)

    avg_s = {k: avg(k) for k in stats_list[0]}
    acc_mean = sum(accs) / len(accs) if accs else None
    acc_std = _statistics.stdev(accs) if len(accs) > 1 else None
    return avg_s, acc_mean, acc_std


def _render_table(col_data):
    headers = col_data[0]
    widths = [max(len(row[i]) for row in col_data) for i in range(len(headers))]
    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    print(sep)
    print('|' + '|'.join(f' {headers[i]:<{widths[i]}} ' for i in range(len(headers))) + '|')
    print(sep)
    for row in col_data[1:]:
        print('|' + '|'.join(f' {row[i]:<{widths[i]}} ' for i in range(len(headers))) + '|')
    print(sep)


def print_table(rows):
    tool_headers = ['System', 'Tool Calls/Q', 'Tools OK %', 'Avg Chars', 'Max Chars']
    score_headers = ['System', 'N', '#Qs', 'Completes', 'Parses', 'EM', 'Acc [%]', 'Failed']

    tool_data = [tool_headers]
    score_data = [score_headers]

    for key, items in _group_rows(rows):
        # Split complete vs incomplete when there are multiple experiments
        if len(items) > 1:
            max_qs = max(s['total_questions'] for _, s, _ in items)
            complete = [(name, s, acc) for name, s, acc in items if s['total_questions'] == max_qs]
            incomplete = [(name, s, acc) for name, s, acc in items if s['total_questions'] != max_qs]
        else:
            complete, incomplete = items, []

        display_groups = [(key, complete)] + [(name, [(name, s, acc)]) for name, s, acc in incomplete]

        for display_key, display_items in display_groups:
            if len(display_items) == 1:
                _, s, acc = display_items[0]
                avg_s = s
                acc_mean, acc_std = acc, None
            else:
                avg_s, acc_mean, acc_std = _avg_stats(display_items)

            n = avg_s['total_questions']
            total = avg_s['total_tool_calls']
            calls_per_q = f"{total / n:.1f}" if n > 0 else 'N/A'
            pct = f"{100 * avg_s['successful_tool_calls'] / total:.1f}%" if total > 0 else 'N/A'
            completes_pct = f"{100 * avg_s['completes'] / n:.0f}%" if n > 0 else 'N/A'
            parses_pct = f"{100 * avg_s['parses'] / n:.0f}%" if n > 0 else 'N/A'
            em_pct = f"{100 * avg_s['em'] / n:.0f}%" if n > 0 else 'N/A'

            if acc_mean is not None:
                acc_str = f"{acc_mean:.1f}" + (f" ± {acc_std:.1f}" if acc_std is not None else "")
            else:
                acc_str = 'N/A'

            tool_data.append([
                display_key,
                calls_per_q,
                pct,
                str(int(avg_s['avg_chars'])),
                str(int(avg_s['max_chars'])),
            ])
            score_data.append([
                display_key,
                str(len(display_items)),
                str(int(n)),
                f"{int(avg_s['completes'])} ({completes_pct})",
                f"{int(avg_s['parses'])} ({parses_pct})",
                f"{int(avg_s['em'])} ({em_pct})",
                acc_str,
                str(int(avg_s['failed'])),
            ])

    _render_table(tool_data)
    print()
    _render_table(score_data)


def print_sample(system_name, rec):
    print(f'\n=== System: {system_name} ===')
    print(json.dumps(rec, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description='Compare BCP evaluation results across multiple systems',
        epilog='Example:\n  %(prog)s --folders results/system_a results/system_b\n'
               '  %(prog)s --folders results/system_a --sample failed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--folders',
        nargs='+',
        required=True,
        help='Folders to analyze. Each folder is treated as one system.',
    )
    parser.add_argument(
        '--sample',
        choices=['failed', 'completes', 'parses', 'not-parses'],
        help='Sample and print one trajectory per system from this category.',
    )
    parser.add_argument(
        '--sample-n',
        type=int,
        default=1,
        metavar='N',
        help='Number of trajectories to sample per system (default: 1).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for sampling.',
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    rows = []
    all_records = {}  # system_name -> records (only loaded when needed)

    for folder in args.folders:
        name = os.path.basename(folder.rstrip('/'))
        records = load_records(folder)
        all_records[name] = records
        stats = analyze_records(records)
        acc = load_accuracy(folder)
        rows.append((name, stats, acc))

    print_table(rows)

    if args.sample:
        print()
        samples = []
        for name, records in all_records.items():
            if args.sample == 'failed':
                pool = [r for r in records if r.get('status', 'success') != 'success']
            elif args.sample == 'completes':
                pool = [r for r in records if ends_with_assistant(r)]
            elif args.sample == 'parses':
                pool = [r for r in records if has_exact_answer(r)]
            else:  # not-parses
                pool = [r for r in records if ends_with_assistant(r) and not has_exact_answer(r)]
            if not pool:
                print(f'[{name}] No records in category "{args.sample}"')
                continue
            chosen = random.sample(pool, min(args.sample_n, len(pool)))
            for rec in chosen:
                samples.append((name, rec))

        for i, (name, rec) in enumerate(samples):
            print_sample(name, rec)
            if i < len(samples) - 1:
                try:
                    import tty
                    import termios
                    print('\n-- Press any key for next sample (Ctrl+C to quit) --', end='', flush=True)
                    with open('/dev/tty') as tty_f:
                        fd = tty_f.fileno()
                        old = termios.tcgetattr(fd)
                        try:
                            tty.setraw(fd)
                            ch = tty_f.read(1)
                        finally:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    if ch == '\x03':  # Ctrl+C
                        raise KeyboardInterrupt
                    print()
                except KeyboardInterrupt:
                    print()
                    break


if __name__ == '__main__':
    main()
