from transformers import pipeline
import spacy
import re

# Chargement des modèles
summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
nlp = spacy.load("fr_core_news_md")

# ========================== FONCTION 1 : Résumé ==========================
def generer_resume(texte: str) -> str:
    try:
        if len(texte) > 2000:
            texte = texte[:2000]
        resume = summarizer(texte, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        return resume.strip()
    except Exception as e:
        return f"❌ Erreur dans le résumé : {str(e)}"

# ========================== FONCTION 2 : Actions ==========================
VERBES_ACTION = [
    "faire", "envoyer", "préparer", "corriger", "mettre", "relancer", "planifier",
    "organiser", "réaliser", "communiquer", "assurer", "valider", "gérer",
    "terminer", "implémenter", "livrer", "analyser", "présenter", "informer"
]

PATTERNS_ACTION = [
    r"\b(doit|devra|faut|à faire|à préparer|à envoyer|penser à)\b[^.]{5,100}[.]",
    r"\b(?:[A-Z][a-z]+)\sdoit\s[^.]+[.]"
]

def extraire_actions_hybride(texte: str) -> list:
    doc = nlp(texte)
    actions = set()

    for sent in doc.sents:
        for token in sent:
            if token.lemma_ in VERBES_ACTION and token.pos_ == "VERB":
                sujets = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubj:pass")]
                if sujets:
                    phrase = sent.text.strip()
                    if len(phrase) > 15:
                        actions.add(phrase.capitalize())

    for pattern in PATTERNS_ACTION:
        for match in re.findall(pattern, texte, flags=re.IGNORECASE):
            phrase = match.strip().capitalize()
            if len(phrase) > 15:
                actions.add(phrase)

    return sorted(actions)

# ========================== FONCTION 3 : Décisions ==========================
PHRASES_DECISION = [
    "nous avons décidé", "il a été décidé", "la décision est", "on a validé",
    "il a été validé", "nous validons", "la direction a approuvé", "l’équipe a choisi",
    "on a choisi de", "il a été convenu", "on retient", "sera retenu", "a été approuvé",
    "go pour", "c’est validé", "on valide", "on est d’accord", "on y va", "accord trouvé",
    "le choix est fait", "d’accord pour", "solution retenue", "c’est acté"
]

PATTERNS_DECISION = [
    r"\b(?:nous avons décidé de|il a été décidé que|la décision est de|on a validé que|il a été validé que|nous validons|on retient que)[^.]{10,100}[.]",
    r"Décision ?[:\-]\s*[^.]+[.]",
    r"\b(?:il a été convenu|il est acté que|on choisit de|c’est validé|on y va|on est d’accord|go pour|le choix est fait|c’est acté|d’accord pour)[^.]{5,100}[.]"
]

def extraire_decisions_hybride(texte: str) -> list:
    doc = nlp(texte)
    decisions = set()

    for sent in doc.sents:
        contenu = sent.text.strip().lower()
        if any(phrase in contenu for phrase in PHRASES_DECISION):
            if len(contenu) > 15:
                decisions.add(sent.text.strip().capitalize())

    for pattern in PATTERNS_DECISION:
        matches = re.findall(pattern, texte, flags=re.IGNORECASE)
        for m in matches:
            phrase = m.strip().capitalize()
            if len(phrase) > 15:
                decisions.add(phrase)

    return sorted(decisions)
