import json




def read_jsonl(file_path):
    i = 0
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            i += 1
            try:
                data.append(json.loads(line))
            except:
                print(i)
    return data


data = read_jsonl('metadata.jsonl')

print(len(data))

print("*" * 80 + "没类型")
for i in data:
    # i['Annex'] = ""
    for j in i['score']:
        if 'type' not in j:
            print(str(i['Level']) + '   ' + i['task_name'] + '   ' + j['question'])

print("*" * 80 + "选择题没选项")
for i in data:
    # i['Annex'] = ""
    for j in i['score']:
        if 'type' in j and j['type'] == "Multiple choice" and "choices" not in j:
            print(str(i['Level']) + '   ' + i['task_name'] + '   ' + j['question'])
