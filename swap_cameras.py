#!/usr/bin/env python3

"""
Kamera-Tausch Script für System180 Demonstrator
===============================================
Dieses Script tauscht die linke und rechte Kamera ganz einfach um.
"""

import os
import re

def read_current_config():
    """Liest die aktuelle Kamera-Konfiguration aus main.py"""
    if not os.path.exists("main.py"):
        print("❌ main.py nicht gefunden!")
        return None, None
    
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Suche nach den Zeilen wo die Seriennummern zugewiesen werden
    left_match = re.search(r'left_oak_serial\s*=\s*oak_serials\[(\d+)\]', content)
    right_match = re.search(r'right_oak_serial\s*=\s*oak_serials\[(\d+)\]', content)
    
    if left_match and right_match:
        left_index = int(left_match.group(1))
        right_index = int(right_match.group(1))
        return left_index, right_index
    
    return None, None

def swap_cameras_in_main():
    """Tauscht die Kamera-Zuordnung in main.py"""
    print("🔄 Tausche Kameras in main.py...")
    
    if not os.path.exists("main.py"):
        print("❌ main.py nicht gefunden!")
        return False
    
    # Backup erstellen
    with open("main.py", "r", encoding="utf-8") as f:
        original_content = f.read()
    
    with open("main.py.backup", "w", encoding="utf-8") as f:
        f.write(original_content)
    print("📋 Backup erstellt: main.py.backup")
    
    # Aktuelle Konfiguration lesen
    left_index, right_index = read_current_config()
    
    if left_index is None or right_index is None:
        print("❌ Konnte aktuelle Kamera-Zuordnung nicht finden!")
        return False
    
    print(f"Aktuelle Zuordnung: Links=Index{left_index}, Rechts=Index{right_index}")
    
    # Tausche die Indizes
    new_content = original_content
    new_content = re.sub(r'left_oak_serial\s*=\s*oak_serials\[\d+\]', 
                        f'left_oak_serial  = oak_serials[{right_index}]', new_content)
    new_content = re.sub(r'right_oak_serial\s*=\s*oak_serials\[\d+\]', 
                        f'right_oak_serial = oak_serials[{left_index}]', new_content)
    
    # Speichere die geänderte Datei
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"✅ Kameras getauscht! Neue Zuordnung: Links=Index{right_index}, Rechts=Index{left_index}")
    return True

def update_camera_config():
    """Aktualisiert die camera_config.py falls vorhanden"""
    if os.path.exists("camera_config.py"):
        print("🔄 Aktualisiere camera_config.py...")
        
        with open("camera_config.py", "r") as f:
            content = f.read()
        
        # Extrahiere die aktuellen Serials
        left_match = re.search(r'LEFT_CAMERA_SERIAL = "([^"]+)"', content)
        right_match = re.search(r'RIGHT_CAMERA_SERIAL = "([^"]+)"', content)
        
        if left_match and right_match:
            left_serial = left_match.group(1)
            right_serial = right_match.group(1)
            
            # Tausche die Serials
            new_content = content.replace(f'LEFT_CAMERA_SERIAL = "{left_serial}"', 
                                        f'LEFT_CAMERA_SERIAL = "{right_serial}"')
            new_content = new_content.replace(f'RIGHT_CAMERA_SERIAL = "{right_serial}"', 
                                            f'RIGHT_CAMERA_SERIAL = "{left_serial}"')
            
            with open("camera_config.py", "w") as f:
                f.write(new_content)
            
            print("✅ camera_config.py aktualisiert")

def main():
    print("🔄 System180 Kamera-Tausch")
    print("=" * 50)
    print()
    
    # Zeige aktuelle Konfiguration
    left_index, right_index = read_current_config()
    
    if left_index is not None and right_index is not None:
        print("Aktuelle Kamera-Zuordnung:")
        print(f"  LINKS  = Kamera mit Index {left_index}")
        print(f"  RECHTS = Kamera mit Index {right_index}")
        print()
        
        print("Nach dem Tausch wird es so sein:")
        print(f"  LINKS  = Kamera mit Index {right_index}")
        print(f"  RECHTS = Kamera mit Index {left_index}")
        print()
        
        print("Möchtest du die Kameras tauschen? (j/n): ", end="")
        answer = input().lower().strip()
        
        if answer in ['j', 'ja', 'y', 'yes']:
            if swap_cameras_in_main():
                update_camera_config()
                print()
                print("🎉 Kameras erfolgreich getauscht!")
                print()
                print("Nächste Schritte:")
                print("1. Starte die Anwendung: bash run.sh")
                print("2. Prüfe ob die Kameras jetzt richtig zugeordnet sind")
                print("3. Falls nicht: Führe dieses Script nochmal aus")
                print()
                print("💡 Tipp: Du kannst 'python3 find_cameras.py' verwenden")
                print("   um die Live-Vorschau zu sehen und zu prüfen")
            else:
                print("❌ Fehler beim Tauschen der Kameras")
        else:
            print("Vorgang abgebrochen.")
    else:
        print("❌ Konnte aktuelle Kamera-Konfiguration nicht lesen!")
        print("Stelle sicher, dass main.py vorhanden und korrekt ist.")

if __name__ == "__main__":
    main()