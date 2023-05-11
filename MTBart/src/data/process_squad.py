import json

with open('dev-v2.0.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

output_data = []

for article in data['data']:
    for p in article['paragraphs']:
        for qas in p['qas']:
            answers = {
                "text": [],
                "answer_start": []
            }
            for ans in qas['answers']:
                answers['text'].append(ans['text'])
                answers['answer_start'].append(ans['answer_start'])

            output_data.append({
                "id": qas['id'],
                "context": p['context'],
                "question": qas['question'],
                "answers": answers
            })
with open('dev_output.json', 'w') as f:
    json.dump({'version': 'v2.0', 'data': output_data}, f)