import unicodedata

content = open('src/modeling/baseline_training.py', 'r', encoding='utf-8').read()

def to_ascii(text):
    result = []
    for c in text:
        if ord(c) < 128:
            result.append(c)
        elif ord(c) == 0x2500:
            result.append('-')
        elif ord(c) == 0x2501:
            result.append('=')
        elif ord(c) == 0x2502:
            result.append('|')
        else:
            norm = unicodedata.normalize('NFKD', c)
            ascii_eq = norm.encode('ascii', 'ignore').decode('ascii')
            result.append(ascii_eq if ascii_eq else '?')
    return ''.join(result)

fixed = to_ascii(content)

open('src/modeling/baseline_training.py', 'w', encoding='utf-8').write(fixed)
remaining = sum(1 for c in fixed if ord(c) > 127)
print(f'Remaining non-ASCII chars: {remaining}')
print('Done - file cleaned.')
