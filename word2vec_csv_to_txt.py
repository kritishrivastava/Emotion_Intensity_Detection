import csv
import os

path = os.getcwd()
direc = os.getcwd()
path += "/data/word2vec/anger.csv"

with open(path, 'r', encoding="utf8") as inf:
    reader = csv.reader(inf)
    write_path = direc + '/data/word2vec/anger.txt'
    with open(write_path, 'w', encoding="utf8") as out:
        writer = csv.writer(out, delimiter='\t')
        writer.writerows(reader)
inf.close()

path = os.getcwd()
path += "/data/word2vec/joy.csv"

with open(path, 'r', encoding="utf8") as inf:
    reader = csv.reader(inf)
    write_path = direc + '/data/word2vec/joy.txt'
    with open(write_path, 'w', encoding="utf8") as out1:
        writer = csv.writer(out1, delimiter='\t')
        writer.writerows(reader)
inf.close()

path = os.getcwd()
path += "/data/word2vec/fear.csv"

with open(path, 'r', encoding="utf8") as inf:
    reader = csv.reader(inf)
    write_path = direc + '/data/word2vec/fear.txt'
    with open(write_path, 'w', encoding="utf8") as out2:
        writer = csv.writer(out2, delimiter='\t')
        writer.writerows(reader)
inf.close()

path = os.getcwd()
path += "/data/word2vec/sadness.csv"

with open(path, 'r', encoding="utf8") as inf:
    reader = csv.reader(inf)
    write_path = direc + '/data/word2vec/sadness.txt'
    with open(write_path, 'w', encoding="utf8") as out3:
        writer = csv.writer(out3, delimiter='\t')
        writer.writerows(reader)
inf.close()