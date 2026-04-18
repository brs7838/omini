import os
import json
import uuid
import shutil
import httpx
import asyncio

SOURCE_DIR = r"C:\Users\pc\Downloads\Voice Lib"
PROJECT_ROOT = r"E:\Ai\Omini"
VOICES_DIR = os.path.join(PROJECT_ROOT, "assets", "voices")
VOICES_JSON = os.path.join(VOICES_DIR, "voices.json")

GENDER_MAP = {
    "alia bhatt": "female",
    "amrish puri": "male",
    "deepika": "female",
    "kader khan": "male",
    "nana patekar": "male",
    "pankaj tripathi": "male",
    "shraddha kapoor": "female",
    "sunil shetty": "male",
    "sunny deol": "male"
}

async def generate_persona(name, gender):
    print(f"Generating persona for {name}...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("http://localhost:8000/voices/generate-persona", 
                                   json={"name": name, "gender": gender})
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        print(f"Failed to generate for {name}: {e}")
    return {"age": "40", "about": "A talented individual.", "catchphrases": "I am ready."}

async def main():
    if not os.path.exists(VOICES_JSON):
        voices = []
    else:
        with open(VOICES_JSON, "r", encoding="utf-8") as f:
            voices = json.load(f)

    existing_names = [v['name'].lower() for v in voices]
    
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".mpeg") or f.endswith(".mp3")]
    
    for filename in files:
        # Clean name from filename
        clean_name = filename.split(".")[0].replace("_", " ").title()
        if clean_name.lower() in existing_names:
            print(f"Skipping {clean_name}, already exists.")
            continue
            
        gender = GENDER_MAP.get(clean_name.lower(), "male")
        voice_id = str(uuid.uuid4())[:8]
        new_filename = f"{voice_id}_{filename.replace(' ', '_')}"
        dest_path = os.path.join(VOICES_DIR, new_filename)
        
        # Copy file
        shutil.copy(os.path.join(SOURCE_DIR, filename), dest_path)
        
        # Get Persona
        persona = await generate_persona(clean_name, gender)
        
        new_voice = {
            "id": voice_id,
            "name": clean_name,
            "gender": gender,
            "style": "cloned",
            "age": persona.get("age", "30"),
            "about": persona.get("about", ""),
            "catchphrases": persona.get("catchphrases", ""),
            "file_path": f"assets/voices/{new_filename}"
        }
        
        voices.append(new_voice)
        print(f"Imported {clean_name}")

    with open(VOICES_JSON, "w", encoding="utf-8") as f:
        json.dump(voices, f, indent=2, ensure_ascii=False)
    
    print("Bulk import complete!")

if __name__ == "__main__":
    asyncio.run(main())
