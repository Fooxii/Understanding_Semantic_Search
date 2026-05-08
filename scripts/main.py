from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Dogs are friendly",
    "Cats are independent",
    "Cars are fast"
]

embeddings = model.encode(sentences)

print(embeddings) #prints embeddings
#print(len(embeddings)) #prints number of embeddings generated
#print(len(embeddings[0])) #prints how many dimensions each embedding has
