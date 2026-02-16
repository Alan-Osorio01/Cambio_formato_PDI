# Color Converter - Procesamiento de Imagenes

Proyecto de clase para conversion de color en imagenes y camara en tiempo real usando OpenCV y C++.

## Autores
- Alan Osorio
- Juan Camilo Gallardo

## Que es este proyecto
Esta aplicacion toma una imagen o una camara como entrada y convierte los frames a uno de estos espacios de color:
- `GRAY` (escala de grises)
- `YUV` (implementado como `YCrCb` en OpenCV)
- `HSV`

La ejecucion es por menu interactivo, asi que no necesitas flags como `--image` o `--mode`.

## Como funciona
El flujo principal es:
1. El usuario ejecuta `./build/app`.
2. La app muestra un menu para elegir fuente (`Imagen` o `Camara`).
3. La app muestra un menu para elegir conversion (`GRAY`, `YUV`, `HSV`).
4. Si se elige imagen:
   - carga archivo con `ImageIO`
   - convierte con `ColorConverter`
   - muestra original y convertido
   - opcionalmente guarda el resultado
5. Si se elige camara:
   - abre el dispositivo con `CameraStream`
   - convierte cada frame en tiempo real
   - termina con `q` o `ESC`

## Estructura del proyecto
```text
converter/
  CMakeLists.txt
  include/
    App.h
    converters/ColorConverter.h
    io/ImageIO.h
    io/CameraStream.h
  src/
    main.cpp
    App.cpp
    converters/ColorConverter.cpp
    io/ImageIO.cpp
    io/CameraStream.cpp
```

## Requisitos (Linux)
- `g++` con soporte C++17
- `cmake` >= 3.16
- OpenCV 4 (`libopencv-dev`)

En Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

## Como compilar en Linux
Desde la carpeta `converter/`:
```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

## Como ejecutar correctamente
Desde `converter/`:
```bash
./build/app
```

Al ejecutar, veras menus como estos:
```text
Selecciona fuente:
  1) Imagen
  2) Camara

Selecciona conversion:
  1) Escala de grises (GRAY)
  2) YUV (YCrCb)
  3) HSV
```

Si eliges `Imagen`, ingresa una ruta valida, por ejemplo:
- `foto.jpeg`
- `./imagenes/foto.jpeg`

Si eliges guardar, ingresa un nombre de salida, por ejemplo:
- `fotogris.png`
- `resultado_hsv.png`

## Notas utiles
- Si `YUV/HSV` se ven con colores raros al mostrarse, es normal: se esta visualizando el espacio convertido directamente.
- Si la camara no abre, prueba otro indice de dispositivo (`0`, `1`, `2`).
- Para ayuda rapida:
```bash
./build/app --help
```

