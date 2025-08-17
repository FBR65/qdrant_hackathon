# Qdrant Hackathon - Bildsuche & Verarbeitung

Ein fortschrittliches Bildsuch- und Verarbeitungs-System, das KI-gestützte Bildanalyse mit Vektorsuche in Qdrant kombiniert.

![Qdrant Logo](project_images/qdrant_model.png)
![System Architektur](project_images/mermaid_diagram_1755449238483.png)

## Funktionen

### Kernfunktionen
- **Bildsuche**: Vektorbasierte Bildsuche mit Text- oder Bild-Query
- **Bildverarbeitung**: Automatische Bildanalyse mit KI
- **Metadaten-Extraktion**: GPS-Koordinaten, Bildbeschreibungen und Tags
- **Bulk-Verarbeitung**: Massenverarbeitung von Bildern aus Verzeichnissen
- **Gradio-Oberfläche**: Benutzerfreundliche Web-Oberfläche mit monochromem Theme

### Technologie-Stack
- **Gradio 5.34**: Web-Oberfläche mit monochromem Theme
- **Qdrant**: Vektordatenbank für Bildsuche
- **Ollama**: KI-Modell für Bildanalyse (mistral-small3.2)
- **CLIP**: Bildvektorisierung (laion/clip-vit-b-32-laion2B-s34B-b79K)
- **EXIF-Verarbeitung**: GPS-Koordinaten und Metadaten

## Projektstruktur

```
qdrant_hackathon/
├── app.py                    # Haupt-Gradio Anwendung
├── main.py                   # Einstiegspunkt
├── config.py                 # Konfiguration
├── image_processor.py        # Haupt-Bildverarbeitung
├── ollama_client.py          # Ollama/OpenAI Client
├── clip_processor.py         # CLIP Bildvektorisierung
├── qdrant_manager.py         # Qdrant Datenbank-Manager
├── utils.py                  # Hilfsfunktionen
├── pyproject.toml            # Projekt-Abhängigkeiten
├── README.md                 # Diese Datei
└── project_images/           # Beispielbilder
```

## Installation

### Voraussetzungen
- Python 3.11+
- Qdrant Server (lokal oder remote)
- Ollama mit mistral-small3.2 Modell

### Setup

1. **Repository klonen**
```bash
git clone <repository-url>
cd qdrant_hackathon
```

2. **Virtuelle Umgebung erstellen**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows
```

3. **Abhängigkeiten installieren**
```bash
uv sync
```

4. **Qdrant Server starten**
```bash
# Lokaler Qdrant Server
docker run -p 6333:6333 qdrant/qdrant
```

5. **Ollama Modell herunterladen**
```bash
ollama pull mistral-small3.2
```

## Starten

### Entwicklung
```bash
uv run python main.py --debug
```

### Produktion
```bash
uv run python main.py --host 0.0.0.0 --port 8504
```

### Konfigurationsoptionen
```bash
uv run python main.py --help
```

Verfügbare Optionen:
- `--host`: Server Host (Standard: 0.0.0.0)
- `--port`: Server Port (Standard: 8504)
- `--share`: Anwendung öffentlich teilen
- `--allowed-paths`: Erlaubte Pfade für Dateizugriff
- `--ollama-url`: Ollama API URL
- `--ollama-model`: Ollama Modellname
- `--qdrant-host`: Qdrant Host
- `--qdrant-port`: Qdrant Port
- `--debug`: Debug-Modus aktivieren

## Bedienung

### Tab 1: Bildsuche
- **Textsuche**: Geben Sie Suchbegriffe ein, um ähnliche Bilder zu finden
- **Bildsuche**: Laden Sie ein Bild hoch, um ähnliche Bilder zu finden
- **Ergebnisanzeige**: Ergebnisse werden mit Ähnlichkeitsscores angezeigt

### Tab 2: Einzelbild-Upload
- **Bild hochladen**: Wählen Sie ein Bild aus Ihrer lokalen Festplatte
- **Automatische Verarbeitung**: Das Bild wird automatisch analysiert
- **Metadaten-Anzeige**: Tags, Beschreibung und GPS-Koordinaten werden angezeigt
- **Datenbank-Speicherung**: Ergebnisse werden in Qdrant gespeichert

### Tab 3: Bulkverarbeitung
- **Verzeichnis-Pfad**: Geben Sie den Pfad zu einem Verzeichnis mit Bildern an
- **Batch-Verarbeitung**: Alle Bilder im Verzeichnis werden automatisch verarbeitet
- **Fortschrittsanzeige**: Echtzeit-Anzeige des Verarbeitungsfortschritts
- **Ergebnisübersicht**: Zusammenfassung der verarbeiteten und fehlgeschlagenen Bilder

## Konfiguration

### Umgebungsvariablen
```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export OLLAMA_BASE_URL=http://localhost:11434/v1/
export OLLAMA_MODEL=mistral-small3.2:latest
export ALLOWED_PATHS=/path/to/images:/another/path
```

### Konfigurationsdatei
In `config.py` können Sie folgende Einstellungen anpassen:
- Qdrant Verbindungseinstellungen
- Ollama Modellkonfiguration
- Unterstützte Bildformate
- Erlaubte Pfade für Dateizugriff
- Server-Einstellungen

## Datenbank-Struktur

### Qdrant Collections
Das System erstellt 4 Collections für verschiedene Distanzmetriken:
- `image_db_cosine`: Kosinus-Ähnlichkeit
- `image_db_euclid`: Euklidische Distanz
- `image_db_dot`: Dot-Produkt
- `image_db_manhattan`: Manhattan-Distanz

### Metadaten-Struktur
Jedes Bild wird mit folgenden Metadaten gespeichert:
```json
{
  "image_id": "uuid",
  "file_path": "/pfad/zum/bild.jpg",
  "file_name": "bild.jpg",
  "file_size": 1024000,
  "width": 1920,
  "height": 1080,
  "format": "JPEG",
  "processing_timestamp": "2024-01-01T12:00:00",
  "gps_coordinates": [52.5200, 13.4050],
  "location_name": "Berlin, Germany",
  "ai_tags": ["stadt", "architektur", "gebäude"],
  "ai_description": "Ein modernes Gebäude in Berlin mit Glasfassade",
  "model_used": "mistral-small3.2:latest",
  "embedding_dim": 512,
  "embedding": [0.1, 0.2, ...],
  "processed_at": "2024-01-01T12:00:00",
  "source_type": "upload"
}
```

## Suchfunktionen

### Textbasierte Suche
- **Semantische Suche**: Suchbegriffe werden in Vektoren umgewandelt
- **Tag-Suche**: Suche nach spezifischen Bild-Tags
- **Ortssuche**: Suche nach Bildern aus bestimmten Regionen

### Bildbasierte Suche
- **Ähnlichkeitssuche**: Hochgeladenes Bild wird vektorisiert
- **Multi-Metric Suche**: Ergebnisse für verschiedene Distanzmetriken
- **Score-basierte Anzeige**: Ähnlichkeitsscores werden angezeigt

## Sicherheit

### Dateizugriff
- **Allowed Paths**: Nur explizit erlaubte Pfade sind zugänglich
- **Pfad-Validierung**: Absolute Pfade werden überprüft
- **Sandboxing**: Zugriff auf Systemdateien ist blockiert

### API-Sicherheit
- **OpenAI Compatibility**: Nutzung von Ollama mit OpenAI API
- **Token-Management**: Sichere Handhabung von API-Schlüsseln
- **Rate Limiting**: Schutz vor Missbrauch

## Fehlerbehebung

### Häufige Probleme

1. **Qdrant Verbindung**
```bash
# Überprüfen Sie Qdrant Verbindung
curl http://localhost:6333/
```

2. **Ollama Verbindung**
```bash
# Überprüfen Sie Ollama Verbindung
curl http://localhost:11434/api/tags
```

3. **Modell-Download**
```bash
# Laden Sie das CLIP Modell herunter
ollama pull laion/clip-vit-b-32-laion2B-s34B-b79K
```

### Debug-Modus
Aktivieren Sie den Debug-Modus für detaillierte Fehlermeldungen:
```bash
uv run python main.py --debug
```

## Performance

### Optimierungen
- **Batch-Verarbeitung**: Mehrere Bilder gleichzeitig verarbeiten
- **Caching**: Zwischenspeichern von Ergebnissen
- **Asynchrone Verarbeitung**: Parallele Verarbeitung von Anfragen

### Skalierung
- **Verteilte Verarbeitung**: Unterstützung für mehrere Worker
- **Lastverteilung**: Load Balancing für Qdrant Cluster
- **Index-Optimierung**: Regelmäßige Optimierung der Vektoren

## Contributing

1. Fork das Repository
2. Erstellen Sie einen Feature-Branch
3. Commit Ihre Änderungen
4. Push Sie den Branch
5. Erstellen Sie einen Pull Request

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Siehe die LICENSE-Datei für weitere Informationen.

## Danksagung

- **Qdrant**: Für die hervorragende Vektordatenbank
- **Ollama**: Für die einfache Nutzung von KI-Modellen
- **Hugging Face**: Für die CLIP Modelle und Transformer-Bibliothek
- **Gradio**: Für die intuitive Web-Oberfläche

## Kontakt

Für Fragen und Unterstützung:
- Erstellen Sie ein Issue im Repository
- Kontaktieren Sie das Entwicklungsteam

---

**Hinweis**: Dieses Projekt wurde im Rahmen des Qdrant Hackathons entwickelt und dient als Demonstrationsplattform für moderne Bildsuch- und Verarbeitungs-Technologien.