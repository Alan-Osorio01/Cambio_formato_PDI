#!/usr/bin/env python3
"""
prepare_selfies_v2.py
Pipeline completo:
  1. Detectar y recortar rostro en cada selfie (Haar cascade)
  2. Data augmentation: rotaciones, escala, flip, brillo
  3. Reentrenar PCA con el binario C++
  4. Copiar assets a Android Studio
"""

import cv2
import numpy as np
import os
import glob
import shutil
import subprocess
import sys

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
FRG_DIR       = os.path.join(BASE_DIR, "FRG")
OUTPUT_DIR    = os.path.join(FRG_DIR, "faces", "alan_aug")
LIST_FILE     = os.path.join(FRG_DIR, "list", "train_list.txt")
BINARY        = os.path.join(FRG_DIR, "build", "frg")
CASCADE_PATH  = os.path.join(FRG_DIR, "haarcascade", "haarcascade_frontalface_default.xml")
SELFIES_DIR   = os.environ.get(
    "SELFIES_DIR", os.path.expanduser("~/Descargas/fotos"))
ASSETS_DIR    = os.path.join(
    BASE_DIR, "android-app", "app", "src", "main", "assets")
PERSON_NAME   = os.environ.get("PERSON_NAME", "Alan")


def detect_face(gray, detector):
    """Detecta el rostro más grande; devuelve el recorte o None."""
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4,
        minSize=(60, 60), maxSize=(900, 900)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    # Expandir el recorte un 15 % para incluir algo de frente/mentón
    pad = int(max(w, h) * 0.15)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)
    return gray[y1:y2, x1:x2]


def augment(face100):
    """Genera 8 variaciones de una cara 100×100 ya en escala de grises."""
    h, w = face100.shape[:2]
    cx, cy = w // 2, h // 2
    BR = cv2.BORDER_REFLECT

    variants = [face100.copy()]

    # Rotaciones
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        variants.append(cv2.warpAffine(face100, M, (w, h), borderMode=BR))

    # Flip horizontal
    variants.append(cv2.flip(face100, 1))

    # Escala centrada 0.9×
    M09 = cv2.getRotationMatrix2D((cx, cy), 0, 0.9)
    variants.append(cv2.warpAffine(face100, M09, (w, h), borderMode=BR))

    # Escala centrada 1.1×
    M11 = cv2.getRotationMatrix2D((cx, cy), 0, 1.1)
    variants.append(cv2.warpAffine(face100, M11, (w, h), borderMode=BR))

    # Brillo –15 %
    variants.append(np.clip(face100.astype(np.float32) * 0.85, 0, 255).astype(np.uint8))

    # Brillo +15 %
    variants.append(np.clip(face100.astype(np.float32) * 1.15, 0, 255).astype(np.uint8))

    return variants   # 10 variantes en total


def main():
    print("=== prepare_selfies_v2.py ===\n")

    for path, label in [(CASCADE_PATH, "Haar cascade"), (BINARY, "Binario C++")]:
        if not os.path.exists(path):
            print(f"ERROR — {label} no encontrado: {path}")
            sys.exit(1)

    detector = cv2.CascadeClassifier(CASCADE_PATH)

    # ── Buscar selfies ───────────────────────────────────────────────────────
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp",
                "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
    selfies = sorted(sum(
        (glob.glob(os.path.join(SELFIES_DIR, p)) for p in patterns), []))

    if not selfies:
        print(f"No se encontraron imágenes en: {SELFIES_DIR}")
        sys.exit(1)

    print(f"Selfies encontradas: {len(selfies)}")

    # ── Limpiar directorio de salida ─────────────────────────────────────────
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # ── Fase 1 + 2: Detectar, recortar, aumentar ────────────────────────────
    saved = []
    no_face = 0
    idx = 0

    for i, path in enumerate(selfies):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [{i+1}] OMITIDO (no se pudo leer): {os.path.basename(path)}")
            continue

        crop = detect_face(img, detector)
        if crop is None:
            no_face += 1
            tag = "sin rostro → imagen completa"
            crop = img
        else:
            tag = f"rostro {crop.shape[1]}×{crop.shape[0]}"

        # Redimensionar + ecualizar
        face100 = cv2.resize(crop, (100, 100))
        cv2.equalizeHist(face100, face100)

        variants = augment(face100)
        for v in variants:
            out = os.path.join(OUTPUT_DIR, f"aug_{idx:04d}.bmp")
            cv2.imwrite(out, v)
            saved.append(out)
            idx += 1

        print(f"  [{i+1}/{len(selfies)}] {os.path.basename(path)} — {tag} → {len(variants)} variantes")

    n_per_img = 10
    print(f"\n✓ {len(saved)} imágenes guardadas "
          f"({len(selfies) - no_face} con rostro + {no_face} sin detección) "
          f"× {n_per_img} variantes")

    # ── Fase 3: Actualizar train_list.txt ────────────────────────────────────
    with open(LIST_FILE, "w") as f:
        for p in saved:
            rel = os.path.relpath(p, FRG_DIR)
            f.write(f"{PERSON_NAME};{rel}\n")
    print(f"✓ train_list.txt → {len(saved)} entradas")

    # ── Fase 4: Reentrenar PCA ───────────────────────────────────────────────
    print("\nEntrenando PCA (modo 1)…")
    result = subprocess.run(
        [BINARY], input="1\n", capture_output=True, text=True, cwd=FRG_DIR)
    if result.stdout.strip():
        print(result.stdout)
    if result.returncode != 0:
        print("ERROR en entrenamiento:\n", result.stderr)
        sys.exit(1)
    print("✓ Entrenamiento completado")

    # ── Fase 5: Copiar assets a Android ─────────────────────────────────────
    os.makedirs(ASSETS_DIR, exist_ok=True)
    data_dir = os.path.join(FRG_DIR, "data")
    print("\nCopiando assets…")
    for fname in ["facesdata.txt", "mean.txt", "eigen.txt"]:
        src = os.path.join(data_dir, fname)
        dst = os.path.join(ASSETS_DIR, fname)
        shutil.copy2(src, dst)
        kb = os.path.getsize(dst) / 1024
        print(f"  {fname}: {kb:.0f} KB")

    print("\n✓ ¡Listo! Ahora haz ▶ Run en Android Studio.")
    print("  Filtra Logcat por PCA_DIST y confirma los valores de Alan vs extraños.")


if __name__ == "__main__":
    main()
