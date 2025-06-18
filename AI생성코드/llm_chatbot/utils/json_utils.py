import json

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_user_preferences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
