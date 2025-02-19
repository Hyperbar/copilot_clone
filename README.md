# copilot_clone
Interactive assistant that can answer questions based on captured images and the user's


**Fonctionnalités du programme :**
- ✅ Capture et affiche la vidéo de la webcam.
- ✅ Écoute en continu la voix et la convertit en texte via Whisper.
- ✅ Envoie la question et une image de la webcam à GPT-4o.
- ✅ Lit la réponse générée à voix haute avec `openai.audio.speech`.
- ✅ Utilise une interaction en temps réel avec l'utilisateur.

C'est un assistant interactif qui peut répondre aux questions en fonction de l'image capturée et de la voix de l'utilisateur.

---

### Pré-requis
Vous devez avoir une clé API `OPENAI_API_KEY` et une clé API `GOOGLE_API_KEY` pour exécuter ce code. Stockez-les dans un fichier `.env` à la racine du projet ou définissez-les comme variables d'environnement.

---

### Étapes d'installation

#### 1. **Installer PortAudio sur Windows**

PortAudio est une bibliothèque utilisée pour gérer l'audio. Sur Windows, il n'est pas nécessaire d'utiliser `brew`. Utilise plutôt une version pré-compilée de PortAudio :

1. Télécharge le fichier `portaudio` adapté à ton système depuis ce lien :  
   [PortAudio - Windows Install](http://portaudio.com/download.html)
   
2. Suis les instructions pour l'installer sur Windows.

#### 2. **Créer un environnement virtuel et installer les dépendances**

Dans le terminal, dans le dossier de ton projet, exécute les commandes suivantes :

- Créer un environnement virtuel :
  ```bash
  python -m venv venv
  ```

- Activer l'environnement virtuel :
  ```bash
  venv\Scripts\activate
  ```

- Mettre à jour `pip` et installer les dépendances :
  ```bash
  pip install -U pip
  pip install -r requirements.txt
  ```

#### 3. **Exécuter l'assistant**

Pour lancer l'assistant, utilise la commande suivante :
```bash
python assistant.py
```

