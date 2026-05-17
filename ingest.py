import json
import os
import chromadb
from chromadb.utils import embedding_functions

print("Initializing Vector Database...")
db_client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = db_client.get_or_create_collection(
    name="shl_assessments",
    embedding_function=sentence_transformer_ef
)

def ingest_catalog():
    print("Loading catalog data...")
    file_path = "shl_catalog.json"
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f, strict=False)
        
    documents = []
    metadatas = []
    ids = []
    
    print(f"Found {len(data)} items in the JSON. Processing...")
    
    for index, item in enumerate(data):
        # 1. Map to your actual JSON keys
        name = item.get("name", "Unknown Test")
        url = item.get("link", "")  # Changed to 'link' based on your data
        
        # 2. 'keys' is a list in your JSON, so we join it into a single string
        keys_list = item.get("keys", [])
        test_type = ", ".join(keys_list) if keys_list else "Unknown"
        
        # 3. Grab the extra data to make the AI smarter
        description = item.get("description", "")
        job_levels = item.get("job_levels_raw", "")
        
        # 4. Create a super-rich document for the AI to search against
        searchable_text = f"Assessment Name: {name}. \nCategories: {test_type}. \nTarget Levels: {job_levels}. \nDescription: {description}"
        
        # 5. Keep the metadata strictly tied to what the SHL schema requires
        metadata = {
            "name": name,
            "url": url,
            "test_type": test_type
        }
        
        documents.append(searchable_text)
        metadatas.append(metadata)
        ids.append(f"test_{index}")
        
    print(f"Inserting {len(documents)} assessments into ChromaDB...")
    
    # Optional: Clear the collection first if you are re-running this script
    # so you don't get duplicate entries
    # collection.delete(ids) 
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print("Success! The catalog has been vectorized and stored.")

if __name__ == "__main__":
    ingest_catalog()