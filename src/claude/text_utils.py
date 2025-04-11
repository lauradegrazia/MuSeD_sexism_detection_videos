import numpy as np
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support
)

def basic_prompt(line):
   
    out = f"""
### Instruction:
Determine if the text is sexist.

### Text:
{line["text"].strip()}

### Is the text sexist?:"""

    return out

def basic_prompt_clean_txt(line):
  
    out = f"""
### Instruction:
Determine if the text is sexist.

### Text:
{line["text_clean"].strip()}

### Is the text sexist?:"""

    return out

# make this option to choose text or text_clean
def basic_prompt_es(line):
    
    out = f"""
### Instrucción:
Determina si el texto es sexista.

### Texto:
{line["text"].strip()}

### Es el texto sexista?:"""

    return out

def basic_prompt_es_clean_txt(line):
    
    out = f"""
### Instrucción:
Determina si el texto es sexista.

### Texto:
{line["text_clean"].strip()}

### Es el texto sexista?:"""

    return out

def basic_video_prompt_es(line):
    out = f"""
### Instrucción:
Analiza el contenido de este video y determina si es sexista. **Si el texto del video es sexista, el video también debe clasificarse como sexista, incluso si las imágenes no parecen ofensivas.** 

### Texto del video:
{line["text_clean"].strip()}

### Pregunta:
Basado en el texto y las imágenes proporcionadas, ¿el video es sexista? Responde solo con 'Sí' o 'No'. 
Si el texto es sexista, responde 'Sí' sin importar las imágenes.
"""

    return out

def basic_parser(input_str, language="en"):
    yes_lang = {"en": "yes", "es": "sí"}
    yes = yes_lang[language]

    if type(input_str) == str:
        input_str = input_str.lower()
        if yes not in input_str and "no" not in input_str:
            return None
        elif yes in input_str and "no" not in input_str:
            return 1
        elif "no" in input_str and yes not in input_str:
            return 0
        elif input_str.find(yes) < input_str.find("no"):
            return 1
        elif input_str.find("no") < input_str.find(yes):
            return 0
        else:
            return None
    else:
        return None


def get_classification_metrics(df, true_col, pred_col):
    y_true = df[true_col]
    y_pred = df[pred_col]
    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_neg": precision[0],
        "precision_pos": precision[1],
        "recall_neg": recall[0],
        "recall_pos": recall[1],
        "f1_neg": f1[0],
        "f1_pos": f1[1],
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    return metrics






