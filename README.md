# 🐍 Snake con Detección de Dedo

Un juego clásico de Snake controlado mediante detección de gestos de mano usando visión por computadora. Mueve tu dedo índice frente a la cámara para controlar la dirección de la serpiente.

## 📋 Descripción

Este proyecto combina el juego clásico Snake con tecnología de detección de gestos usando MediaPipe y OpenCV. En lugar de usar el teclado, puedes controlar la serpiente moviendo tu dedo índice frente a la cámara web.

## ✨ Características

- 🎮 Control mediante detección de gestos de mano
- 📹 Visualización en tiempo real de la detección de manos
- 🎨 Interfaz moderna con colores vibrantes
- 📊 Sistema de puntuación y seguimiento de longitud
- 🔄 Reinicio fácil del juego
- ⚡ Velocidad de juego ajustable

## 🛠️ Tecnologías Utilizadas

- **Python 3.12+**
- **OpenCV** - Procesamiento de video y visión por computadora
- **MediaPipe** - Detección de manos y seguimiento de landmarks
- **Pygame** - Motor de juego y renderizado
- **NumPy** - Operaciones matemáticas y arrays

## 📦 Requisitos

- Python 3.12 o superior
- Cámara web conectada y funcionando
- Windows, Linux o macOS

## 🚀 Instalación

1. **Clona o descarga este repositorio**

2. **Instala las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

   Esto instalará automáticamente:
   - `opencv-python==4.8.1.78`
   - `mediapipe>=0.10.7`
   - `pygame==2.5.2`
   - `numpy>=1.24.3`

## 🎯 Cómo Ejecutar

1. **Asegúrate de que tu cámara web esté conectada y funcionando**

2. **Ejecuta el juego:**

   ```bash
   python snake.py
   ```

3. **El juego se iniciará y verás:**
   - Una ventana de Pygame con el juego Snake
   - Una ventana de OpenCV mostrando el feed de tu cámara con la detección de manos

## 🎮 Cómo Jugar

1. **Posiciona tu mano frente a la cámara** - Asegúrate de que tu mano sea claramente visible

2. **Mueve tu dedo índice** para controlar la dirección de la serpiente:
   - **Mueve el dedo hacia la derecha** → La serpiente va a la derecha
   - **Mueve el dedo hacia la izquierda** → La serpiente va a la izquierda
   - **Mueve el dedo hacia arriba** → La serpiente va hacia arriba
   - **Mueve el dedo hacia abajo** → La serpiente va hacia abajo

3. **Come la comida roja** para crecer y aumentar tu puntuación (+10 puntos por comida)

4. **Evita chocar** con:
   - Los bordes de la pantalla
   - Tu propio cuerpo

## ⌨️ Controles del Teclado

- **R** - Reiniciar el juego (cuando hay Game Over)
- **ESC** - Salir del juego
- **Q** - Cerrar la ventana de la cámara y salir

## 📊 Información en Pantalla

El juego muestra:
- **Score**: Tu puntuación actual
- **Longitud**: Longitud actual de la serpiente
- **Dirección**: Dirección actual detectada
- **FPS**: Frames por segundo del juego
- **Estado**: Si está esperando detección de mano o mostrando la dirección

## ⚙️ Configuración

Puedes ajustar las siguientes variables en `snake.py`:

- `GAME_SPEED` (línea 38): Velocidad del juego (FPS). Valor por defecto: 8
- `WINDOW_WIDTH` y `WINDOW_HEIGHT` (líneas 22-23): Tamaño de la ventana del juego
- `min_detection_confidence` y `min_tracking_confidence` (líneas 14-15): Sensibilidad de la detección de manos

## 🐛 Solución de Problemas

### La cámara no se abre
- Verifica que tu cámara esté conectada
- Asegúrate de que no esté siendo usada por otra aplicación
- En algunos sistemas, puede ser necesario cambiar el índice de la cámara en la línea 199: `cv2.VideoCapture(0)` a `cv2.VideoCapture(1)`

### La detección de manos no funciona bien
- Asegúrate de tener buena iluminación
- Mantén tu mano a una distancia adecuada de la cámara
- Evita fondos muy complejos o similares al color de tu piel

### El juego va muy rápido o muy lento
- Ajusta la variable `GAME_SPEED` en el código (línea 38)
- Valores más bajos = más lento
- Valores más altos = más rápido

## 📝 Notas

- El juego está optimizado para detectar una sola mano a la vez
- La detección funciona mejor con buena iluminación y fondo contrastante
- El dedo índice debe estar claramente visible para un control preciso

## 📄 Licencia

Ver el archivo LICENSE para más detalles.

## 👨‍💻 Autor

Proyecto desarrollado como demostración de integración de visión por computadora con desarrollo de juegos.

---

¡Disfruta jugando Snake con tus gestos! 🎮✨

