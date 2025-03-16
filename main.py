import streamlit as st
from utils import (
    load_model,
    extract_text_from_pdf,
    clean_extracted_text,
    split_into_paragraphs,
    predict_clauses,
    label_mapping
)

model, tokenizer = load_model()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une page :", ["Accueil", "Extraction de Clauses"])

# Page d'Accueil
if page == "Accueil":
    st.title("LifeInsurance - ClauseFinder")
    st.write("""
    ### Bienvenue sur **LifeInsurance - ClauseFinder** !
    Cette application vous permet d'analyser les conditions générales de vos contrats d'assurance vie et d'identifier automatiquement les clauses importantes.

    #### Fonctionnalités :
    - **Extraction de Texte** : Extrait le texte des fichiers PDF.
    - **Identification des Clauses** : Utilise un modèle FlauBERT fine-tuné pour identifier les clauses importantes.
    - **Affichage des Résultats** : Affiche les clauses importantes de manière claire et organisée.

    #### Comment Utiliser l'Application :
    1. Cliquez sur **Extraction de Clauses** dans la barre latérale.
    2. Téléversez un fichier PDF contenant les conditions générales de votre contrat.
    3. L'application analysera le document et affichera les clauses importantes.

    #### À Propos :
    Cette application a été développée pour simplifier l'analyse des contrats d'assurance vie aux clients. Elle utilise des technologies avancées de traitement du langage naturel (NLP) pour fournir des résultats précis et fiables.
    """)

# Page d'Extraction de Clauses
elif page == "Extraction de Clauses":
    st.title("Extraction de Clauses Importantes")
    
    
    pdf_path = st.file_uploader(
        "Veuillez téléverser les conditions générales de contrat d'assurance vie en version PDF",
        type=".pdf"
    )

    if pdf_path:
        
        st.info("Extraction du texte PDF en cours...")
        text = extract_text_from_pdf(pdf_path)
        if not text:
            st.error("Aucun texte extrait du PDF.")
            st.stop()  

        text = clean_extracted_text(text)

        
        paragraphs = split_into_paragraphs(text)

        
        important_paragraphs = []
        st.info("Prédiction des clauses importantes en cours...")
        progress_bar = st.progress(0)  
        for i, paragraph in enumerate(paragraphs):
            try:
                prediction = predict_clauses(paragraph, model, tokenizer)
                category = label_mapping.get(prediction, "inconnu")

                
                if prediction == 1:
                    important_paragraphs.append(paragraph)
            except Exception as e:
                st.error(f"Erreur lors de la prédiction de la clause : {e}")

            
            progress_bar.progress((i + 1) / len(paragraphs))

        
        st.success(f"{len(important_paragraphs)} clauses importantes trouvées.")
        if important_paragraphs:
            st.write("### Clauses importantes :")
            for i, clause in enumerate(important_paragraphs, 1):
                st.write(f"**Clause {i} :**")
                st.write(clause)
                st.write("---")  
        else:
            st.warning("Aucune clause importante n'a été détectée.")