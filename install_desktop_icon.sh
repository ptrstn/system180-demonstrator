#!/bin/bash

# ==============================================================================
# Desktop Icon Installer für System180 Demonstrator
# ==============================================================================
# Installiert ein Desktop Icon zum einfachen Starten der Anwendung
# ==============================================================================

echo "🖥️  System180 Desktop Icon Installer"
echo "====================================================="
echo ""

# Prüfe ob Desktop Ordner existiert
DESKTOP_DIR="$HOME/Desktop"
if [ ! -d "$DESKTOP_DIR" ]; then
    echo "❌ Desktop Ordner nicht gefunden: $DESKTOP_DIR"
    echo "   Erstelle Desktop Ordner..."
    mkdir -p "$DESKTOP_DIR"
fi

# Prüfe ob system180-demonstrator Ordner am richtigen Ort ist
PROJECT_DIR="$HOME/Desktop/system180-demonstrator"
CURRENT_DIR=$(pwd)

echo "🔍 Überprüfe Projekt-Pfad..."
if [ "$CURRENT_DIR" != "$PROJECT_DIR" ]; then
    echo "⚠️  Aktueller Pfad: $CURRENT_DIR"
    echo "⚠️  Erwarteter Pfad: $PROJECT_DIR"
    echo ""
    echo "Das Desktop Icon erwartet das Projekt unter:"
    echo "  ~/Desktop/system180-demonstrator"
    echo ""
    echo "Optionen:"
    echo "1. Projekt dorthin verschieben/kopieren"
    echo "2. Desktop Icon für aktuellen Pfad anpassen"
    echo ""
    echo "Soll ich das Icon für den aktuellen Pfad anpassen? (j/n): "
    read -r answer
    
    if [[ "$answer" == "j" || "$answer" == "ja" || "$answer" == "y" || "$answer" == "yes" ]]; then
        PROJECT_DIR="$CURRENT_DIR"
        echo "✅ Verwende aktuellen Pfad: $PROJECT_DIR"
    else
        echo "❌ Installation abgebrochen."
        echo "   Verschiebe dein Projekt nach ~/Desktop/system180-demonstrator"
        echo "   und führe das Script erneut aus."
        exit 1
    fi
fi

# Erstelle Desktop-Datei mit korrektem Pfad
DESKTOP_FILE="$DESKTOP_DIR/System180-Demonstrator.desktop"

echo "📝 Erstelle Desktop Icon..."
cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=System180 Demonstrator
Comment=Startet den System180 Demonstrator mit 3-Kamera Setup
Exec=gnome-terminal --working-directory="$PROJECT_DIR" --title="System180 Demonstrator" -- bash -c './run.sh; echo ""; echo "Drücke Enter zum Schließen..."; read'
Icon=applications-utilities
Terminal=false
StartupNotify=true
Categories=Development;Utility;
Keywords=camera;yolo;detection;system180;
StartupWMClass=gnome-terminal-server
Path=$PROJECT_DIR
EOF

# Mache Desktop-Datei ausführbar
chmod +x "$DESKTOP_FILE"

# Versuche auch das Icon vertrauenswürdig zu machen (Ubuntu 22.04+)
if command -v gio &> /dev/null; then
    gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
fi

echo "✅ Desktop Icon erstellt: $DESKTOP_FILE"
echo ""

# Überprüfe ob run.sh existiert und ausführbar ist
if [ -f "$PROJECT_DIR/run.sh" ]; then
    if [ -x "$PROJECT_DIR/run.sh" ]; then
        echo "✅ run.sh gefunden und ausführbar"
    else
        echo "⚠️  run.sh gefunden, aber nicht ausführbar"
        echo "   Führe aus: chmod +x run.sh"
    fi
else
    echo "❌ run.sh nicht gefunden in: $PROJECT_DIR"
    echo "   Stelle sicher, dass alle Scripts vorhanden sind"
fi

echo ""
echo "🎉 Installation abgeschlossen!"
echo ""
echo "📋 Verwendung:"
echo "1. Gehe zu deinem Desktop"
echo "2. Doppelklick auf 'System180 Demonstrator'"
echo "3. Ein Terminal öffnet sich und startet die Anwendung"
echo "4. Browser öffnet automatisch http://localhost:8000"
echo ""
echo "💡 Tipps:"
echo "- Das Terminal bleibt offen bis du Enter drückst"
echo "- Bei Fehlern siehst du die Meldungen im Terminal"
echo "- Zum Beenden: Strg+C im Terminal drücken"
echo ""

# Bonus: Erstelle auch Starter für andere wichtige Scripts
echo "🔧 Möchtest du auch Icons für andere Scripts erstellen?"
echo "   - Kamera Setup (find_cameras.py)"
echo "   - System Info (jetson_info.py)" 
echo "   - Model Converter (convert_models_to_engine.sh)"
echo ""
echo "Zusätzliche Icons erstellen? (j/n): "
read -r create_more

if [[ "$create_more" == "j" || "$create_more" == "ja" || "$create_more" == "y" || "$create_more" == "yes" ]]; then
    
    # Kamera Setup Icon
    cat > "$DESKTOP_DIR/System180-Camera-Setup.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=System180 Camera Setup
Comment=Kamera Setup und Identifikation für System180
Exec=gnome-terminal --working-directory="$PROJECT_DIR" --title="System180 Camera Setup" -- bash -c 'python3 snippets/find_cameras.py; echo ""; echo "Drücke Enter zum Schließen..."; read'
Icon=camera-web
Terminal=false
StartupNotify=true
Categories=Development;Utility;
Path=$PROJECT_DIR
EOF
    chmod +x "$DESKTOP_DIR/System180-Camera-Setup.desktop"
    
    # System Info Icon
    cat > "$DESKTOP_DIR/System180-System-Info.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=System180 System Info
Comment=Zeigt Jetson System-Informationen an
Exec=gnome-terminal --working-directory="$PROJECT_DIR" --title="System180 System Info" -- bash -c 'python3 snippets/jetson_info.py; echo ""; echo "Drücke Enter zum Schließen..."; read'
Icon=computer
Terminal=false
StartupNotify=true
Categories=Development;Utility;
Path=$PROJECT_DIR
EOF
    chmod +x "$DESKTOP_DIR/System180-System-Info.desktop"
    
    # Model Converter Icon
    cat > "$DESKTOP_DIR/System180-Convert-Models.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=System180 Convert Models
Comment=Konvertiert YOLO Modelle zu TensorRT Engines
Exec=gnome-terminal --working-directory="$PROJECT_DIR" --title="System180 Model Converter" -- bash -c './convert_models_to_engine.sh; echo ""; echo "Drücke Enter zum Schließen..."; read'
Icon=applications-development
Terminal=false
StartupNotify=true
Categories=Development;Utility;
Path=$PROJECT_DIR
EOF
    chmod +x "$DESKTOP_DIR/System180-Convert-Models.desktop"
    
    echo "✅ Zusätzliche Icons erstellt:"
    echo "   - System180 Camera Setup"
    echo "   - System180 System Info" 
    echo "   - System180 Convert Models"
fi

echo ""
echo "🎯 Alle Desktop Icons sind bereit!"
echo "   Schaue auf deinem Desktop nach den neuen Icons."