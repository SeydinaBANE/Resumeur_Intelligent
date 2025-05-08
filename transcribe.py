import whisper
import os

# Chargement du modèle Whisper une seule fois
try:
    model = whisper.load_model("base")  # nécessite openai-whisper
except Exception as e:
    model = None
    print(f"❌ Erreur lors du chargement du modèle Whisper : {str(e)}")

def transcrire_audio(audio_path: str, save_txt: bool = True) -> str:
    """
    Transcrit un fichier audio avec Whisper (OpenAI).
    Sauvegarde facultative de la transcription au format texte.
    """
    if model is None:
        return "❌ Modèle Whisper non chargé."

    if not os.path.isfile(audio_path):
        return f"❌ Fichier introuvable : {audio_path}"

    try:
        print(f"📥 Transcription de : {audio_path}")
        result = model.transcribe(audio_path, fp16=False)

        texte = result.get("text", "").strip()
        if not texte:
            return "❌ Aucun texte détecté."

        if save_txt:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = f"{base_name}_transcription.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(texte)

        print("✅ Transcription terminée.")
        return texte

    except Exception as e:
        return f"❌ Erreur lors de la transcription : {str(e)}"
