import gradio as gr
import tempfile
import os

from transcribe import transcrire_audio
from extract import (
    generer_resume,
    extraire_actions_hybride,
    extraire_decisions_hybride
)

def analyser_reunion(fichier_audio):
    if not fichier_audio:
        return "❌ Aucun fichier fourni.", "", "", None

    # Étape 1 : Transcription de l'audio
    transcription = transcrire_audio(fichier_audio.name)
    if transcription.startswith("❌"):
        return transcription, "", "", None

    # Étape 2 : Analyse NLP
    resume = generer_resume(transcription)
    actions = extraire_actions_hybride(transcription)
    decisions = extraire_decisions_hybride(transcription)

    # Étape 3 : Formatage des résultats
    resume_affichage = f"📄 Résumé :\n{resume}"
    actions_affichage = (
        "📝 Tâches identifiées :\n" + "\n".join([f"- {a}" for a in actions])
        if actions else "Aucune tâche détectée."
    )
    decisions_affichage = (
        "✅ Décisions prises :\n" + "\n".join([f"- {d}" for d in decisions])
        if decisions else "Aucune décision détectée."
    )

    # Étape 4 : Export dans un fichier texte
    contenu_export = (
        f"{resume_affichage}\n\n"
        f"{actions_affichage}\n\n"
        f"{decisions_affichage}\n\n"
        f"🗣️ Transcription brute :\n{transcription}"
    )

    nom_fichier = "compte_rendu_reunion.txt"
    chemin_export = os.path.join(tempfile.gettempdir(), nom_fichier)
    with open(chemin_export, "w", encoding="utf-8") as f:
        f.write(contenu_export)

    return resume_affichage, actions_affichage, decisions_affichage, chemin_export

# Interface Gradio
demo = gr.Interface(
    fn=analyser_reunion,
    inputs=[
        gr.File(label="🎧 Fichier audio/vidéo de la réunion", file_types=[".mp3", ".mp4", ".wav", ".m4a"])
    ],
    outputs=[
        gr.Textbox(label="📌 Résumé synthétique"),
        gr.Textbox(label="📝 Liste des tâches"),
        gr.Textbox(label="✅ Liste des décisions"),
        gr.File(label="⬇️ Télécharger le compte-rendu complet")
    ],
    title="AI Résumeur de Réunion",
    description=(
        "Uploade un enregistrement de réunion et récupère automatiquement un résumé, "
        "les décisions prises et les tâches à effectuer."
    ),
)

if __name__ == "__main__":
    demo.launch()
