from gensim.models import Word2Vec

inputFilePath = "./Data/WashedData.txt"
embeddingSize = 256
weightOutputPath = "./Data/weightMatrix.txt"

corporas = []
with open(inputFilePath,"r",encoding="UTF-8") as rh:
    for line in rh:
        oneLine = line.strip()
        sentence = oneLine.split("\t")[0].strip()
        corporas.append(sentence.split())
        print(sentence.split()[-1])
print("Data has been read.")
print(corporas[150])
print("Training...")
model = Word2Vec(corporas,size = embeddingSize,window=7, min_count=1,workers=12, iter=120)
with open(weightOutputPath,"w",encoding="UTF-8") as wh:
    for word in model.wv.index2entity:
        vec = model.wv[word]
        wh.write(word + "\t")
        print(word)
        for number in vec:
            wh.write(str(number) + ",")
        wh.write("\n")



