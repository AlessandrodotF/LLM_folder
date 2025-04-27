from FlagEmbedding import BGEM3FlagModel
#2024
#100 lingue, 8192 tokes
#addestramento: self-knoledge distillation, Architettura: XLM-Roberta
#utilizza rappresentazioni dense per confrontare query e documenti.
#emula metodi come BM25, basandosi su rappresentazioni sparse.
#impiega più vettori per rappresentare un singolo documento, migliorando la precisione nel matching fine-grained.​
model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

embeddings_1 = model.encode(sentences_1, 
                            batch_size=12, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']
similarity = embeddings_1 @ embeddings_2.T
print(similarity)