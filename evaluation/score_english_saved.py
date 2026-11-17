"""Score saved English answers without rerunning retrieval or generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from run_full_nojev import CHAT, append, load_env, read_jsonl

PROMPT = ('Judge the factual correctness of an answer to the given question against '
          'the reference answers. Accept equivalent names and a correct answer embedded '
          'in a sentence. Reject a wrong entity, missing requested fact, contradiction, '
          'or alternatives that include incorrect answers. Do not reward keyword overlap. '
          'Return only a JSON object with correct (boolean) and reason (short string).')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dataset', choices=['musique', '2wikimultihopqa'], required=True)
    a = p.parse_args()
    folder = a.run / a.dataset
    answers = read_jsonl(folder / 'answers.jsonl')
    if not answers:
        raise ValueError('No saved answers')
    load_env()
    from openai import OpenAI
    client = OpenAI(api_key=os.environ['VGRAG_CHAT_KEY'], base_url='https://api.siliconflow.cn/v1',
                    timeout=90, max_retries=2)
    done = {r['qid']: r for r in read_jsonl(folder / 'semantic-scores.jsonl')}
    for row in answers:
        payload = {'question': row['question'], 'references': row['gold_answers'],
                   'prediction': row['prediction']}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if row['qid'] in done:
            if done[row['qid']]['input_sha256'] != fingerprint:
                raise ValueError('Saved answer changed')
            continue
        response = client.chat.completions.create(
            model=CHAT, messages=[{'role': 'system', 'content': PROMPT},
                                 {'role': 'user', 'content': json.dumps(payload)}],
            response_format={'type': 'json_object'}, temperature=0, max_tokens=1000,
            extra_body={'enable_thinking': False})
        append(folder / 'judge-usage.jsonl', {'qid': row['qid'], 'model': CHAT,
            'finish_reason': response.choices[0].finish_reason,
            'usage': response.usage.model_dump() if response.usage else None})
        if response.choices[0].finish_reason != 'stop':
            raise ValueError('Judge output truncated')
        grade = json.loads(response.choices[0].message.content)
        if type(grade.get('correct')) is not bool or not isinstance(grade.get('reason'), str):
            raise ValueError('Invalid judge output')
        record = {'qid': row['qid'], 'model': CHAT, 'input_sha256': fingerprint, **grade}
        append(folder / 'semantic-scores.jsonl', record)
        done[row['qid']] = record
        print(a.dataset, 'graded', len(done), '/', len(answers), flush=True)
    summary = {'dataset': a.dataset, 'n': len(answers), 'semantic_judge_model': CHAT,
        'semantic_correct': sum(done[r['qid']]['correct'] for r in answers),
        'semantic_accuracy': sum(done[r['qid']]['correct'] for r in answers) / len(answers),
        'recall_at_5': sum(len({p['id'] for p in r['result']['passages'][:5]} & set(r['gold_ids'])) /
                           len(r['gold_ids']) for r in answers) / len(answers),
        'support_recall': sum(r['support_recall'] for r in answers) / len(answers),
        'exact_match_runner': sum(r['exact_match'] for r in answers) / len(answers),
        'f1_runner': sum(r['f1'] for r in answers) / len(answers),
        'scope': 'Shared union of sampled questions passages; not full official corpus. '
                 'Runner EM/F1 normalization is not the official dataset scorer. '
                 'Semantic accuracy is a model judgement, not human verification.'}
    (folder / 'audited-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    client.close()


if __name__ == '__main__':
    main()
