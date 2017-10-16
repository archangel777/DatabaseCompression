import random

sample_size = 0

columns = [3, 4, 5, 6, 7]

with open('lineitem.csv', 'r') as fr:
	with open('sampled_items.csv', 'w') as fw:
		sample = fr.readlines() if sample_size == 0 else random.sample(fr.readlines(), sample_size)
		for line in sample:
			splt = line.split('|')
			v = []
			for i in columns:
				v.append(splt[i])
			newline = '|'.join(v) + '\n'
			fw.write(newline)
			

	
