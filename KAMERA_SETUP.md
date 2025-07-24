# 📹 Kamera Setup Guide - Für Nicht-Programmierer

**Problem:** Du weißt nicht, welche OAK-Kamera physisch links oder rechts steht!

**Lösung:** Diese 2 einfachen Scripts helfen dir dabei.

---

## 🔍 Schritt 1: Kameras identifizieren

Führe diesen Befehl aus:
```bash
python3 find_cameras.py
```

**Was passiert:**
- Das Script zeigt dir alle gefundenen Kameras an
- Du siehst Live-Vorschau von beiden Kameras **gleichzeitig**
- Jedes Fenster zeigt an, ob es gerade als "LINKS" oder "RECHTS" verwendet wird
- Du kannst sehen, welche physische Kamera welche Rolle hat

**Beispiel-Ausgabe:**
```
🔍 Suche nach OAK-Kameras...
✅ 2 OAK-Kamera(s) gefunden:

  Kamera 1:
    Serial: 14442C10D13DAFB900
    Name: OAK-1-Max

  Kamera 2:
    Serial: 14442C10D13DAFB901  
    Name: OAK-1-Max
```

**In der Live-Vorschau siehst du:**
- Fenster 1: "Diese Kamera ist aktuell: LINKS"
- Fenster 2: "Diese Kamera ist aktuell: RECHTS"

---

## 🔄 Schritt 2: Kameras tauschen (falls nötig)

**Falls die Zuordnung falsch ist**, führe aus:
```bash
python3 swap_cameras.py
```

**Was passiert:**
- Das Script zeigt dir die aktuelle Zuordnung
- Du kannst mit "j" (ja) bestätigen, dass du tauschen willst
- Die Kameras werden automatisch getauscht
- Ein Backup wird erstellt (main.py.backup)

**Beispiel-Dialog:**
```
Aktuelle Kamera-Zuordnung:
  LINKS  = Kamera mit Index 0
  RECHTS = Kamera mit Index 1

Nach dem Tausch wird es so sein:
  LINKS  = Kamera mit Index 1
  RECHTS = Kamera mit Index 0

Möchtest du die Kameras tauschen? (j/n): j
✅ Kameras erfolgreich getauscht!
```

---

## 🎯 Kompletter Workflow

### Wenn du die Kameras zum ersten Mal einrichtest:

```bash
# 1. Kameras anschließen (beide OAK-1 Max)
# 2. Kameras identifizieren
python3 find_cameras.py

# 3. Live-Vorschau anschauen (mit 'j' bestätigen)
# 4. Prüfen ob Links/Rechts stimmt
# 5. Falls NEIN: Kameras tauschen
python3 swap_cameras.py

# 6. System starten
bash run.sh
```

### Wenn du später die Zuordnung ändern willst:

```bash
# Einfach tauschen
python3 swap_cameras.py

# System neu starten
bash run.sh
```

---

## 🛠️ Problemlösung

### "Keine OAK-Kameras gefunden"
- **Prüfe:** Sind beide Kameras angeschlossen?
- **Prüfe:** Sind die USB-Kabel richtig eingesteckt?
- **Lösung:** Kameras ab- und wieder anstecken

### "Weniger als zwei OAK-Kameras gefunden"
- **Prüfe:** Beide Kameras müssen angeschlossen sein
- **Prüfe:** Verwende verschiedene USB-Ports
- **Lösung:** Warte 10 Sekunden nach dem Anschließen

### Live-Vorschau zeigt schwarzes Bild
- **Prüfe:** Ist die Kamera-Linse abgedeckt?
- **Prüfe:** Haben die Kameras genug Strom? (USB 3.0 verwenden)
- **Lösung:** Kameras einzeln testen

### Script sagt "main.py nicht gefunden"
- **Lösung:** Stelle sicher, dass du im richtigen Ordner bist
- **Befehl:** `ls` - du solltest main.py sehen können

---

## 💡 Tipps

- **Etikett:** Klebe kleine Zettel auf die Kameras ("L" und "R")
- **Position:** Die linke Kamera sollte physisch links vom zu prüfenden Objekt stehen
- **Test:** Nach dem Tauschen einmal `bash run.sh` ausführen und im Browser prüfen
- **Backup:** Das Script erstellt automatisch Backups, keine Sorge!

---

## 🔙 Rückgängig machen

Falls etwas schief geht:
```bash
# Backup wiederherstellen
cp main.py.backup main.py

# Oder einfach nochmal tauschen
python3 swap_cameras.py
```

**Das Tauschen ist vollständig umkehrbar!** 🔄