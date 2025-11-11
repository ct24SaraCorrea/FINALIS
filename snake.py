import cv2
import mediapipe as mp
import pygame
import numpy as np
import sys
from collections import deque

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Colores modernos
BACKGROUND = (15, 15, 25)
GRID_COLOR = (30, 30, 40)
SNAKE_HEAD = (0, 255, 150)
SNAKE_BODY = (0, 200, 120)
FOOD_COLOR = (255, 50, 50)
TEXT_COLOR = (255, 255, 255)
ACCENT_COLOR = (100, 150, 255)

# Velocidad reducida
GAME_SPEED = 8  # FPS del juego (muy lento)

# Crear ventana
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("🐍 Snake con Detección de Dedo - Ultra Cool")
clock = pygame.time.Clock()

# Fuente
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

class Snake:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.body = deque([(GRID_WIDTH // 2, GRID_HEIGHT // 2)])
        self.direction = (1, 0)  # Derecha
        self.grow = False
    
    def move(self):
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # Verificar colisiones con bordes
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            return False
        
        # Verificar colisión consigo mismo (excluyendo la cola que se va a eliminar)
        body_to_check = list(self.body)
        if not self.grow and len(body_to_check) > 1:
            body_to_check = body_to_check[:-1]  # Excluir la cola que se eliminará
        
        if new_head in body_to_check:
            return False
        
        self.body.appendleft(new_head)
        
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
        
        return True
    
    def change_direction(self, new_dir):
        # Prevenir movimiento inverso
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir
    
    def eat_food(self):
        self.grow = True
    
    def draw(self, surface):
        for i, (x, y) in enumerate(self.body):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            if i == 0:  # Cabeza
                pygame.draw.rect(surface, SNAKE_HEAD, rect)
                pygame.draw.rect(surface, (255, 255, 255), rect, 2)
            else:  # Cuerpo
                pygame.draw.rect(surface, SNAKE_BODY, rect)
                pygame.draw.rect(surface, (0, 150, 100), rect, 1)

class Food:
    def __init__(self):
        self.position = self.generate_position()
    
    def generate_position(self, snake_body=None):
        """Genera una posición aleatoria, evitando el cuerpo de la serpiente"""
        max_attempts = 1000  # Máximo de intentos para evitar loops infinitos
        attempts = 0
        
        while attempts < max_attempts:
            pos = (np.random.randint(0, GRID_WIDTH), np.random.randint(0, GRID_HEIGHT))
            if snake_body is None or pos not in snake_body:
                return pos
            attempts += 1
        
        # Si no se encuentra posición después de muchos intentos, buscar manualmente
        if snake_body is not None:
            for x in range(GRID_WIDTH):
                for y in range(GRID_HEIGHT):
                    if (x, y) not in snake_body:
                        return (x, y)
        
        # Último recurso: posición aleatoria
        return (np.random.randint(0, GRID_WIDTH), np.random.randint(0, GRID_HEIGHT))
    
    def draw(self, surface):
        x, y = self.position
        rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, FOOD_COLOR, rect)
        pygame.draw.rect(surface, (255, 100, 100), rect, 2)
        # Efecto de brillo
        center = (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2)
        pygame.draw.circle(surface, (255, 200, 200), center, GRID_SIZE // 3)

def detect_direction(landmarks, prev_index_pos):
    """
    Detecta la dirección del movimiento del dedo índice
    """
    if landmarks is None:
        return None, prev_index_pos
    
    # Obtener posición del dedo índice (punto 8)
    index_tip = landmarks.landmark[8]
    current_pos = np.array([index_tip.x, index_tip.y])
    
    if prev_index_pos is None:
        return None, current_pos
    
    # Calcular movimiento
    movement = current_pos - prev_index_pos
    threshold = 0.02  # Sensibilidad
    
    # Determinar dirección dominante
    abs_movement = np.abs(movement)
    if np.max(abs_movement) < threshold:
        return None, current_pos
    
    if abs_movement[0] > abs_movement[1]:  # Movimiento horizontal
        if movement[0] > 0:
            return (1, 0), current_pos  # Derecha
        else:
            return (-1, 0), current_pos  # Izquierda
    else:  # Movimiento vertical
        if movement[1] > 0:
            return (0, 1), current_pos  # Abajo
        else:
            return (0, -1), current_pos  # Arriba

def draw_grid(surface):
    """Dibuja una cuadrícula sutil"""
    for x in range(0, WINDOW_WIDTH, GRID_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (WINDOW_WIDTH, y))

def draw_hand_info(surface, direction_text, fps):
    """Dibuja información de la mano y FPS"""
    # Fondo semitransparente
    info_surface = pygame.Surface((300, 100))
    info_surface.set_alpha(180)
    info_surface.fill((0, 0, 0))
    surface.blit(info_surface, (10, 10))
    
    # Texto de dirección
    dir_text = small_font.render(f"Dirección: {direction_text}", True, ACCENT_COLOR)
    surface.blit(dir_text, (20, 20))
    
    # FPS
    fps_text = small_font.render(f"FPS: {fps:.1f}", True, TEXT_COLOR)
    surface.blit(fps_text, (20, 45))
    
    # Instrucciones
    inst_text = small_font.render("Mueve tu dedo índice", True, TEXT_COLOR)
    surface.blit(inst_text, (20, 70))

def main():
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: No se puede abrir la cámara")
        print("💡 Verifica que tu cámara esté conectada y no esté siendo usada por otra aplicación")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    snake = Snake()
    food = Food()
    score = 0
    game_over = False
    prev_index_pos = None
    direction_text = "Esperando..."
    
    # Asegurar que la comida no esté en la serpiente
    food.position = food.generate_position(snake.body)
    
    frame_counter = 0
    
    print("🎮 Iniciando Snake con Detección de Dedo...")
    print("📹 Asegúrate de que tu cámara esté encendida")
    print("👆 Mueve tu dedo índice para controlar la serpiente")
    
    running = True
    while running:
        # Procesar eventos de Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_over:
                    snake.reset()
                    food.position = food.generate_position(snake.body)
                    score = 0
                    game_over = False
                    prev_index_pos = None
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Leer frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se puede leer la cámara. Verifica que esté conectada.")
            continue
        
        # Voltear frame horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar manos
        results = hands.process(rgb_frame)
        
        # Dibujar detecciones en el frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar landmarks de la mano
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
                
                # Resaltar el dedo índice con un círculo más grande
                index_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                cv2.circle(frame, (cx, cy), 15, (0, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 20, (255, 255, 0), 3)
        
        # Detectar dirección del dedo índice
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            direction, prev_index_pos = detect_direction(hand_landmarks, prev_index_pos)
            
            if direction:
                snake.change_direction(direction)
                # Actualizar texto de dirección
                dir_map = {
                    (1, 0): "➡️ Derecha",
                    (-1, 0): "⬅️ Izquierda",
                    (0, 1): "⬇️ Abajo",
                    (0, -1): "⬆️ Arriba"
                }
                direction_text = dir_map.get(direction, "Desconocido")
        else:
            prev_index_pos = None
            direction_text = "Esperando mano..."
        
        # Mostrar texto en la cámara
        cv2.putText(frame, f"Direccion: {direction_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Mueve tu dedo indice", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar ventana de la cámara
        cv2.imshow("📹 Camara - Mueve tu dedo indice", frame)
        
        # Cerrar ventana de cámara con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Actualizar juego solo a velocidad reducida
        if not game_over:
            frame_counter += 1
            if frame_counter >= (60 // GAME_SPEED):  # Control de velocidad
                frame_counter = 0
                
                if not snake.move():
                    game_over = True
                else:
                    # Verificar si comió la comida
                    if snake.body[0] == food.position:
                        snake.eat_food()
                        score += 10
                        # Generar nueva comida evitando el cuerpo de la serpiente
                        food.position = food.generate_position(snake.body)
        
        # Dibujar
        screen.fill(BACKGROUND)
        draw_grid(screen)
        
        if not game_over:
            food.draw(screen)
            snake.draw(screen)
        else:
            # Pantalla de game over
            game_over_text = font.render("GAME OVER", True, (255, 50, 50))
            score_text = font.render(f"Puntuación: {score}", True, TEXT_COLOR)
            restart_text = small_font.render("Presiona R para reiniciar", True, ACCENT_COLOR)
            
            text_rect = game_over_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
            score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
            
            screen.blit(game_over_text, text_rect)
            screen.blit(score_text, score_rect)
            screen.blit(restart_text, restart_rect)
        
        # Mostrar puntuación y longitud de la serpiente
        score_display = font.render(f"Score: {score}", True, TEXT_COLOR)
        length_display = small_font.render(f"Longitud: {len(snake.body)}", True, ACCENT_COLOR)
        screen.blit(score_display, (WINDOW_WIDTH - 150, 20))
        screen.blit(length_display, (WINDOW_WIDTH - 150, 55))
        
        # Información de la mano
        fps = clock.get_fps()
        draw_hand_info(screen, direction_text, fps)
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS para suavidad visual
    
    # Limpiar
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

