import json
import os

TRAINING_DATA_FILE = 'face_embeddings.json'

print("🔍 Inspecting face_embeddings.json...")

if not os.path.exists(TRAINING_DATA_FILE):
    print("❌ File not found!")
    exit(1)

with open(TRAINING_DATA_FILE, 'r') as f:
    data = json.load(f)

print(f"\n📊 Database Structure:")
print(f"  Names: {len(data.get('names', []))}")
print(f"  Categories: {len(data.get('categories', []))}")
print(f"  Embeddings: {len(data.get('embeddings', []))}")
print(f"  Is Trained: {data.get('is_trained', False)}")

print(f"\n👤 People in database:")
for i, name in enumerate(data.get('names', [])):
    emb = data['embeddings'][i]
    cat = data['categories'][i]
    
    # Check embedding structure
    if isinstance(emb, list):
        if len(emb) > 0:
            if isinstance(emb[0], list):
                # List of embeddings
                print(f"  {i+1}. {name} ({cat}): {len(emb)} embeddings")
                for j, e in enumerate(emb):
                    print(f"     - Embedding {j+1}: {len(e)} dimensions")
            else:
                # Single embedding as list
                print(f"  {i+1}. {name} ({cat}): 1 embedding with {len(emb)} dimensions")
        else:
            print(f"  {i+1}. {name} ({cat}): EMPTY embeddings list")
    else:
        print(f"  {i+1}. {name} ({cat}): Invalid type: {type(emb)}")

print("\n" + "="*60)
print("🔧 Options:")
print("  1. Delete file and recapture (recommended)")
print("  2. Keep file and debug further")
print("="*60)

choice = input("\nDelete face_embeddings.json? (y/n): ").strip().lower()

if choice == 'y':
    os.remove(TRAINING_DATA_FILE)
    print("✓ Deleted face_embeddings.json")
    print("  Now restart the app and recapture faces")
else:
    print("  File kept. Check the output above for issues.")