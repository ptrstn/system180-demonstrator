# ==============================================================================
# Make All Scripts Executable (Source Script)
# ==============================================================================
# Macht alle Scripts ausführbar, damit sie mit bash/python ausgeführt werden können
# WICHTIG: Ausführen mit: source make_executable.sh oder . make_executable.sh
# NICHT mit: bash make_executable.sh
# ==============================================================================

echo "🔧 Mache alle Scripts ausführbar..."
echo ""

# Liste aller Scripts die ausführbar gemacht werden sollen
SCRIPTS=(
    "setup.sh"
    "convert_models_to_engine.sh" 
    "run.sh"
    "jetson_info.sh"
    "make_executable.sh"
    "install_desktop_icon.sh"
    "requirements.sh"
    "swap_cameras.py"
    "snippets/find_cameras.py"
    "snippets/jetson_info.py"
    "snippets/print_tensorrt_version.py"
)

# Zähler für erfolgreich geänderte Dateien
SUCCESS_COUNT=0
TOTAL_COUNT=0

echo "Verarbeite folgende Scripts:"
echo "========================================"

for script in "${SCRIPTS[@]}"; do
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    
    if [ -f "$script" ]; then
        # Prüfe aktuelle Berechtigungen
        CURRENT_PERMS=$(stat -c "%a" "$script" 2>/dev/null || echo "???")
        
        # Mache ausführbar
        chmod +x "$script"
        
        if [ $? -eq 0 ]; then
            NEW_PERMS=$(stat -c "%a" "$script" 2>/dev/null || echo "???")
            echo "✅ $script ($CURRENT_PERMS → $NEW_PERMS)"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "❌ $script (Fehler beim chmod)"
        fi
    else
        echo "⚠️  $script (Datei nicht gefunden)"
    fi
done

echo ""
echo "========================================"
echo "📊 Zusammenfassung:"
echo "   Verarbeitet: $TOTAL_COUNT Scripts"
echo "   Erfolgreich: $SUCCESS_COUNT Scripts"
echo "   Fehler: $((TOTAL_COUNT - SUCCESS_COUNT)) Scripts"
echo ""

# Zusätzlich: Prüfe ob alle wichtigen Scripts existieren und ausführbar sind
echo "🔍 Finale Überprüfung:"
echo "========================================"

MAIN_SCRIPTS=("setup.sh" "convert_models_to_engine.sh" "run.sh" "snippets/find_cameras.py" "jetson_info.sh" "install_desktop_icon.sh")

for script in "${MAIN_SCRIPTS[@]}"; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        echo "✅ $script (bereit zum Ausführen)"
    elif [ -f "$script" ]; then
        echo "⚠️  $script (existiert, aber nicht ausführbar)"
    else
        echo "❌ $script (nicht gefunden)"
    fi
done

echo ""
echo "🎉 Fertig! Du kannst jetzt alle Scripts ausführen:"
echo ""
echo "   bash setup.sh"
echo "   bash convert_models_to_engine.sh"
echo "   bash run.sh"
echo "   bash jetson_info.sh"
echo "   bash install_desktop_icon.sh"
echo "   python3 snippets/find_cameras.py"
echo "   python3 swap_cameras.py"
echo "   python3 snippets/jetson_info.py"
echo "   python3 snippets/print_tensorrt_version.py"
echo ""

# Bonus: Zeige auch Python Scripts an (die brauchen kein chmod, aber zur Info)
echo "💡 Weitere Python Scripts:"
if ls *.py >/dev/null 2>&1; then
    for py_file in *.py; do
        if [ -f "$py_file" ]; then
            echo "   python3 $py_file"
        fi
    done
fi

if ls snippets/*.py >/dev/null 2>&1; then
    for py_file in snippets/*.py; do
        if [ -f "$py_file" ]; then
            echo "   python3 $py_file"
        fi
    done
fi
echo ""

echo "✅ Alle Scripts sind jetzt bereit!"