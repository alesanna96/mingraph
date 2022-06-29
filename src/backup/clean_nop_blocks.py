from itertools import groupby
with open('/home/fra/Documents/thesis/src/0a3e80dc7b5a8444a0579ee7af15004219d1ab7fa446d0bc1ab6a6c588f5b597-code','r') as raw:
    lines=[line[line.index(' ')+1:] for line in raw]

print(len([k for k, g in groupby(lines) if 'nop' not in k ]))