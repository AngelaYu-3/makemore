words = open('other/names.txt','r').read().splitlines()
print(words[:10])
print(len(words))
print(min(len(w) for w in words))
print(max(len(w) for w in words))