#!/usr/bin/env python3

"""
Kamera-Finder Script für System180 Demonstrator
================================================
Dieses Script hilft dir zu sehen, welche Kamera-IDs erkannt werden
und zeigt Live-Vorschau von allen OAK-Kameras gleichzeitig.
"""

import depthai as dai
import cv2
import time

def find_oak_cameras():
    """Findet alle verfügbaren OAK-Kameras und zeigt ihre IDs"""
    print("🔍 Suche nach OAK-Kameras...")
    print("=" * 50)
    
    devices = dai.Device.getAllAvailableDevices()
    
    if len(devices) == 0:
        print("❌ Keine OAK-Kameras gefunden!")
        print("   - Sind die Kameras angeschlossen?")
        print("   - Sind die USB-Kabel richtig eingesteckt?")
        return []
    
    print(f"✅ {len(devices)} OAK-Kamera(s) gefunden:")
    print()
    
    for i, device in enumerate(devices):
        serial = device.getMxId()
        name = device.getDeviceName()
        print(f"  Kamera {i+1}:")
        print(f"    Serial: {serial}")
        print(f"    Name: {name}")
        print()
    
    return [device.getMxId() for device in devices]

def show_camera_preview(camera_serials):
    """Zeigt Live-Vorschau aller Kameras gleichzeitig"""
    if len(camera_serials) < 2:
        print("❌ Mindestens 2 Kameras werden benötigt!")
        return
    
    print("📹 Starte Kamera-Vorschau...")
    print("=" * 50)
    print("WICHTIG:")
    print("- Schaue dir die Fenster an und merke dir welche Kamera wo steht")
    print("- Drücke 'q' um die Vorschau zu beenden")
    print("- Drücke 's' um die aktuelle Zuordnung zu speichern")
    print()
    
    cameras = {}
    pipelines = {}
    
    # Erstelle Pipeline für jede Kamera
    for i, serial in enumerate(camera_serials[:2]):  # Nur erste 2 Kameras
        pipeline = dai.Pipeline()
        color_cam = pipeline.createColorCamera()
        color_cam.setPreviewSize(640, 480)
        color_cam.setInterleaved(False)
        
        xlink_out = pipeline.createXLinkOut()
        xlink_out.setStreamName(f"color_{i}")
        color_cam.preview.link(xlink_out.input)
        
        # Verbinde zu spezifischer Kamera
        all_devices = dai.Device.getAllAvailableDevices()
        matching = [d for d in all_devices if d.getMxId() == serial]
        device = dai.Device(pipeline, matching[0])
        
        cameras[i] = {
            'device': device,
            'queue': device.getOutputQueue(f"color_{i}", maxSize=4, blocking=False),
            'serial': serial,
            'window_name': f"Kamera {i+1} (Serial: {serial[-6:]})"
        }
        pipelines[i] = pipeline
    
    print("Kamera-Zuordnung:")
    print(f"  Kamera 1 = {camera_serials[0]} (wird aktuell als LINKS verwendet)")
    print(f"  Kamera 2 = {camera_serials[1]} (wird aktuell als RECHTS verwendet)")
    print()
    
    try:
        while True:
            for i, cam_info in cameras.items():
                queue = cam_info['queue']
                window_name = cam_info['window_name']
                
                img_frame = queue.tryGet()
                if img_frame is not None:
                    frame = img_frame.getCvFrame()
                    
                    # Füge Text hinzu
                    position_text = "LINKS" if i == 0 else "RECHTS"
                    cv2.putText(frame, f"Diese Kamera ist aktuell: {position_text}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Serial: {cam_info['serial']}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, "Drücke 'q' zum Beenden, 's' zum Speichern", 
                              (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_camera_config(camera_serials)
                print("✅ Kamera-Konfiguration gespeichert!")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Vorschau beendet")
    
    finally:
        cv2.destroyAllWindows()
        for cam_info in cameras.values():
            cam_info['device'].close()

def save_camera_config(camera_serials):
    """Speichert die aktuelle Kamera-Zuordnung in eine Datei"""
    config_content = f"""# Kamera-Konfiguration für System180 Demonstrator
# Diese Datei wurde automatisch erstellt
# 
# Aktuelle Zuordnung:
# LINKS  = {camera_serials[0]}
# RECHTS = {camera_serials[1]}
#
# Um die Kameras zu vertauschen, führe aus: python3 swap_cameras.py

LEFT_CAMERA_SERIAL = "{camera_serials[0]}"
RIGHT_CAMERA_SERIAL = "{camera_serials[1]}"
"""
    
    with open("camera_config.py", "w") as f:
        f.write(config_content)
    
    print(f"Konfiguration gespeichert in: camera_config.py")

def main():
    print("🎥 System180 Kamera-Finder")
    print("=" * 50)
    print()
    
    # Finde alle Kameras
    camera_serials = find_oak_cameras()
    
    if len(camera_serials) >= 2:
        print("Möchtest du die Live-Vorschau sehen? (j/n): ", end="")
        answer = input().lower().strip()
        
        if answer in ['j', 'ja', 'y', 'yes', '']:
            show_camera_preview(camera_serials)
        else:
            save_camera_config(camera_serials)
    
    print()
    print("✅ Fertig!")
    print()
    print("Nächste Schritte:")
    print("1. Wenn die Zuordnung stimmt: bash run.sh")
    print("2. Wenn du die Kameras tauschen willst: python3 swap_cameras.py")

if __name__ == "__main__":
    main()