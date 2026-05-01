import json
import os

def load_backtest_results(filepath="backtest_results.json"):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)

def save_prediction(filepath, data):
    # Appends prediction to a persistence file
    history = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                history = json.load(f)
            except:
                pass
    history.append(data)
    with open(filepath, "w") as f:
        json.dump(history, f, indent=4)
        
def load_predictions(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        try:
            return json.load(f)
        except:
            return []
