#!/usr/bin/env python3
"""
Convierte tus selfies a 100x100 BMP en escala de grises,
actualiza train_list.txt con el nombre que ingreses,
y regenera los archivos de entrenamiento llamando al binario C++.

Uso:
  1. Pon tus selfies (JPG/PNG) en la carpeta que te pida
  2. Ejecuta: python3 prepare_selfies.py
"""

import cv2
import os
import sys
import glob
import shutil
import subprocess

FRG_DIR = os.path.join(os.path.dirname(__file__), "FRG")
FACES_DIR = os.path.join(FRG_DIR, "faces")
LIST_FILE = os.path.join(FRG_DIR, "list", "train_list.txt")
BINARY   = os.path.join(FRG_DIR, "build", "frg")

def prepare(selfies_folder: str, person_name: str):
    # Crear carpeta destino para la persona
    out_dir = os.path.join(FACES_DIR, person_name.lower())
    os.makedirs(out_dir, exist_ok=True)

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(selfies_folder, p))
    files = sorted(files)

    if not files:
        print(f"No se encontraron imágenes en: {selfies_folder}")
        sys.exit(1)

    print(f"Procesando {len(files)} imágenes para '{person_name}'...")
    saved = []
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Saltando {f} (no se pudo leer)")
            continue
        img = cv2.resize(img, (100, 100))
        out_path = os.path.join(out_dir, f"s{i+1}.bmp")
        cv2.imwrite(out_path, img)
        saved.append(out_path)
        print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} → {out_path}")

    print(f"\n{len(saved)} imágenes guardadas en {out_dir}")

    # Regenerar train_list.txt (solo con las imágenes de esta persona)
    # Si quieres mantener otras personas, comenta las siguientes líneas y
    # agrega manualmente las entradas al final del archivo.
    with open(LIST_FILE, "w") as lst:
        for path in saved:
            rel = os.path.relpath(path, FRG_DIR)
            lst.write(f"{person_name};{rel}\n")
    print(f"train_list.txt actualizado con {len(saved)} entradas para '{person_name}'")

    # Ejecutar entrenamiento
    print("\nEjecutando entrenamiento (modo 1)...")
    result = subprocess.run(
        [BINARY],
        input="1\n",
        capture_output=True,
        text=True,
        cwd=FRG_DIR
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr)
        sys.exit(1)

    print("\n✓ Entrenamiento completado.")
    print("\nAhora copia estos archivos a la carpeta de assets de Android:")
    data_dir = os.path.join(FRG_DIR, "data")
    assets_dir = os.path.expanduser(
        "~/AndroidStudioProjects/FaceRecognition/app/src/main/assets/"
    )
    for f in ["facesdata.txt", "mean.txt", "eigen.txt"]:
        src = os.path.join(data_dir, f)
        dst = os.path.join(assets_dir, f)
        shutil.copy2(src, dst)
        print(f"  Copiado: {src} → {dst}")
    print("\n✓ Assets actualizados. Haz Run en Android Studio.")


if __name__ == "__main__":
    print("=== Preparar selfies para FaceRecognition PCA ===\n")
    selfies_folder = input("Ruta de tu carpeta de selfies: ").strip()
    if not os.path.isdir(selfies_folder):
        print(f"Carpeta no encontrada: {selfies_folder}")
        sys.exit(1)
    person_name = input("Nombre de la persona (ej: Alan): ").strip()
    if not person_name:
        print("El nombre no puede estar vacío")
        sys.exit(1)
    prepare(selfies_folder, person_name)
