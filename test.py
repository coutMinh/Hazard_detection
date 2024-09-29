import json
a = {1:2, 2:3}
a.update({3:[4]})
a[3].append(2)
with open('check.json', 'w') as fw:
    json.dump(a, fw)