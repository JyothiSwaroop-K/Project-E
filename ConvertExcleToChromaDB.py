# import pandas as pd
# import chromadb 
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# # Load Excel file
# df = pd.read_excel("questions.xlsx")

# # Extract columns
# questions = df["Questions"].astype(str).tolist()
# answers = df["Answer"].astype(str).tolist()
# difficulties = df["Diffuculty"].astype(str).tolist()

# # Initialize ChromaDB client
# embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# client = chromadb.Client()

# # Create or reset collection
# collection = client.create_collection(
#     name="python_rag_questions",
#     embedding_function=embedding_function
# )

# # Insert questions with answers and difficulty as metadata
# collection.add(
#     documents=questions,
#     metadatas=[{"answer": a, "difficulty": d} for a, d in zip(answers, difficulties)],
#     ids=[f"q_{i}" for i in range(len(questions))]
# )

# print(f"✅ Successfully imported {len(questions)} questions into ChromaDB.")




# import_to_chroma.py
# import pandas as pd
# import chromadb
# from chromadb.utils import embedding_functions

# # Initialize ChromaDB client
# client = chromadb.PersistentClient(path="python_quiz_db1")

# # Create collection with embedding function
# sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="all-MiniLM-L6-v2"
# )

# collection = client.create_collection(
#     name="python_questions",
#     embedding_function=sentence_transformer_ef,
#     metadata={"hnsw:space": "cosine"}  # Better for semantic similarity
# )

# # Read Excel file
# df = pd.read_excel("questions.xlsx")

# # Prepare documents for ChromaDB
# documents = []
# metadatas = []
# ids = []

# for _, row in df.iterrows():
#     documents.append(row['Questions'])
#     metadatas.append({
#         "answer": row['Answer'],
#         "difficulty": row['Diffuculty'],
#         "id": str(row['ID'])
#     })
#     ids.append(str(row['ID']))

# # Add to ChromaDB
# collection.add(
#     documents=documents,
#     metadatas=metadatas,
#     ids=ids
# )

# print(f"Successfully added {len(ids)} questions to ChromaDB")



import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="python_quiz_db")

# Create collection with embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="python_questions",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}  # Better for semantic similarity
)

# Read Excel file
df = pd.read_excel("questions.xlsx")

# Prepare documents for ChromaDB
documents = []
metadatas = []
ids = []

for _, row in df.iterrows():
    # Convert all values to string and handle NaN values
    question = str(row['Questions']) if pd.notna(row['Questions']) else ""
    answer = str(row['Answer']) if pd.notna(row['Answer']) else ""
    difficulty = str(row['Difficulty']) if pd.notna(row['Difficulty']) else "Beginner"
    id_val = str(row['ID']) if pd.notna(row['ID']) else str(_)  # Use index if ID is NaN
    
    documents.append(question)
    metadatas.append({
        "answer": answer,
        "difficulty": difficulty,
        "id": id_val
    })
    ids.append(id_val)

# Add to ChromaDB
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Successfully added {len(ids)} questions to ChromaDB")