#!/usr/bin/env python3
"""
Clean face data and prepare for retraining with proper ArcFace alignment
"""
import os
import json
import shutil

print("="*60)
print("🧹 CLEANING FACE DATA FOR RETRAINING")
print("="*60)

# 1. Backup existing face_data.json
if os.path.exists("face_data.json"):
    backup_name = "face_data.json.backup_before_alignment"
    shutil.copy("face_data.json", backup_name)
    print(f"✓ Backed up face_data.json to {backup_name}")

# 2. Delete face_data.json
if os.path.exists("face_data.json"):
    os.remove("face_data.json")
    print("✓ Deleted face_data.json")

# 3. Create fresh empty database
fresh_db = {
    'embeddings': [],
    'names': [],
    'categories': [],
    'is_trained': False
}

with open("face_data.json", "w") as f:
    json.dump(fresh_db, f)
print("✓ Created fresh face_data.json")

# 4. Backup and remove known_faces directory
if os.path.exists("known_faces"):
    backup_name = "known_faces_backup_before_alignment"
    if os.path.exists(backup_name):
        shutil.rmtree(backup_name)
    shutil.move("known_faces", backup_name)
    print(f"✓ Moved known_faces to {backup_name}")

# 5. Create fresh known_faces directory structure
os.makedirs("known_faces/students", exist_ok=True)
os.makedirs("known_faces/parents", exist_ok=True)
print("✓ Created fresh known_faces directory")

print("="*60)
print("✅ CLEANUP COMPLETE")
print("="*60)
print("\nNext steps:")
print("1. Start the server: python run_prod.py")
print("2. Go to Dataset tab")
print("3. Capture 20-30 frames per person")
print("4. Go to Train tab and click Train")
print("5. Go to Recognition tab and test")
print("\nExpected results after retraining:")
print("- Same person: similarity > 0.75")
print("- Different person: similarity < 0.4")
print("="*60)
