from transformers import FlaubertForSequenceClassification, FlaubertTokenizer
import torch
import pymupdf
import re
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource 
def load_model():
    model = FlaubertForSequenceClassification.from_pretrained("fine-tuned-flaubert")
    tokenizer = FlaubertTokenizer.from_pretrained("fine-tuned-flaubert")
    return model, tokenizer

model, tokenizer = load_model()

label_mapping = {
    0: "Autres",
    1: "Clause importante"
}

def extract_text_from_pdf(uploaded_file):
    """
    Extrait le texte d'un fichier PDF téléversé via Streamlit.
    """
    text = ''
    try:
        # Lire le fichier téléversé avec PyMuPDF
        doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text() + "\n\n"  # Ajouter un saut de ligne entre les pages
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction du PDF : {e}")
    return text

def clean_extracted_text(text):
    """
    Nettoie le texte en supprimant les espaces multiples, les sauts de ligne inutiles et en conservant les paragraphes.
    """
    # Supprimer les espaces multiples
    text = re.sub(r' +', ' ', text)
    # Supprimer les sauts de ligne multiples
    text = re.sub(r'\n+', '\n', text)
    # Supprimer les espaces en début et fin de ligne
    text = '\n'.join([line.strip() for line in text.split('\n')])
    return text.strip()

def split_into_paragraphs(text):
    """
    Divise le texte en paragraphes en regroupant les lignes en fonction de leur contenu.
    """
    paragraphs = []
    current_paragraph = []

    for line in text.split('\n'):
        if line.strip():  # Si la ligne n'est pas vide
            current_paragraph.append(line.strip())
        else:  # Si la ligne est vide, fin du paragraphe
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []

    if current_paragraph:  
        paragraphs.append(" ".join(current_paragraph))

    return paragraphs

def preprocess_text(text, tokenizer, max_length=512):
    """
    Tokenize et prétraite le texte pour le modèle.
    """
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoding

def predict_clauses(text, model, tokenizer):
    """
    Prédit la catégorie d'une clause à l'aide du modèle fine-tuné.
    """
    inputs = preprocess_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).item()
    return predictions