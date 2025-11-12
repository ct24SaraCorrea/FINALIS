import cv2
import mediapipe as mp
import pygame
import numpy as np
import sys
from collections import deque, Counter

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Reducido para detectar más fácilmente
    min_tracking_confidence=0.3     # Reducido para mejor seguimiento
)

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Zona del header (área protegida donde no puede aparecer comida ni serpiente)
HEADER_HEIGHT = 3  # Altura del header en unidades de grid (3 filas)
GAME_AREA_START_Y = HEADER_HEIGHT  # Inicio del área de juego

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
pygame.display.set_caption("Snake con Deteccion de Dedo - Ultra Cool")
clock = pygame.time.Clock()

# Fuente
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

class Snake:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Iniciar la serpiente en el área de juego (no en el header)
        start_y = max(GAME_AREA_START_Y + 2, GRID_HEIGHT // 2)
        self.body = deque([(GRID_WIDTH // 2, start_y)])
        self.direction = (1, 0)  # Derecha
        self.grow = False
    
    def move(self):
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # Verificar colisiones con bordes y zona del header
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < GAME_AREA_START_Y or new_head[1] >= GRID_HEIGHT):
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
        """Genera una posición aleatoria, evitando el cuerpo de la serpiente y el header"""
        max_attempts = 1000  # Máximo de intentos para evitar loops infinitos
        attempts = 0
        
        while attempts < max_attempts:
            # Evitar la zona del header (primeras HEADER_HEIGHT filas)
            pos = (np.random.randint(0, GRID_WIDTH), 
                   np.random.randint(GAME_AREA_START_Y, GRID_HEIGHT))
            if snake_body is None or pos not in snake_body:
                return pos
            attempts += 1
        
        # Si no se encuentra posición después de muchos intentos, buscar manualmente
        if snake_body is not None:
            for x in range(GRID_WIDTH):
                for y in range(GAME_AREA_START_Y, GRID_HEIGHT):
                    if (x, y) not in snake_body:
                        return (x, y)
        
        # Último recurso: posición aleatoria en área de juego
        return (np.random.randint(0, GRID_WIDTH), 
                np.random.randint(GAME_AREA_START_Y, GRID_HEIGHT))
    
    def draw(self, surface):
        x, y = self.position
        rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, FOOD_COLOR, rect)
        pygame.draw.rect(surface, (255, 100, 100), rect, 2)
        # Efecto de brillo
        center = (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2)
        pygame.draw.circle(surface, (255, 200, 200), center, GRID_SIZE // 3)

def detect_direction(landmarks, prev_index_pos, direction_history=None):
    """
    Detecta la dirección del movimiento del dedo índice con suavizado
    """
    if landmarks is None:
        return None, prev_index_pos, direction_history
    
    # Obtener posición del dedo índice (punto 8)
    index_tip = landmarks.landmark[8]
    current_pos = np.array([index_tip.x, index_tip.y])
    
    if prev_index_pos is None:
        return None, current_pos, []
    
    # Calcular movimiento
    movement = current_pos - prev_index_pos
    threshold = 0.015  # Sensibilidad aumentada (reducido de 0.02)
    
    # Determinar dirección dominante
    abs_movement = np.abs(movement)
    if np.max(abs_movement) < threshold:
        return None, current_pos, direction_history if direction_history else []
    
    # Determinar dirección
    if abs_movement[0] > abs_movement[1]:  # Movimiento horizontal
        if movement[0] > 0:
            detected_dir = (1, 0)  # Derecha
        else:
            detected_dir = (-1, 0)  # Izquierda
    else:  # Movimiento vertical
        if movement[1] > 0:
            detected_dir = (0, 1)  # Abajo
        else:
            detected_dir = (0, -1)  # Arriba
    
    # Sistema de suavizado: mantener historial de últimas 3 direcciones
    if direction_history is None:
        direction_history = []
    
    direction_history.append(detected_dir)
    if len(direction_history) > 3:
        direction_history.pop(0)
    
    # Si hay al menos 2 direcciones iguales en el historial, usar esa dirección
    if len(direction_history) >= 2:
        # Contar ocurrencias de cada dirección
        dir_counts = Counter(direction_history)
        most_common_dir, count = dir_counts.most_common(1)[0]
        
        # Si la dirección más común aparece al menos 2 veces, usarla
        if count >= 2:
            return most_common_dir, current_pos, direction_history
    
    # Si no hay consenso, usar la última dirección detectada
    return detected_dir, current_pos, direction_history

def draw_grid(surface):
    """Dibuja una cuadrícula sutil"""
    for x in range(0, WINDOW_WIDTH, GRID_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (WINDOW_WIDTH, y))

def draw_header(surface, direction_text, fps, score, snake_length):
    """Dibuja el header con toda la información del juego"""
    # Fondo del header (barra superior completa)
    header_height = HEADER_HEIGHT * GRID_SIZE
    header_surface = pygame.Surface((WINDOW_WIDTH, header_height))
    header_surface.set_alpha(230)
    header_surface.fill((20, 20, 30))
    surface.blit(header_surface, (0, 0))
    
    # Línea divisoria entre header y área de juego
    pygame.draw.line(surface, GRID_COLOR, (0, header_height), (WINDOW_WIDTH, header_height), 2)
    
    # Información izquierda: Dirección y FPS
    dir_text = small_font.render(f"Direccion: {direction_text}", True, ACCENT_COLOR)
    surface.blit(dir_text, (10, 5))
    
    fps_text = small_font.render(f"FPS: {fps:.1f}", True, TEXT_COLOR)
    surface.blit(fps_text, (10, 25))
    
    # Información central: Instrucciones
    inst_text = small_font.render("Mueve tu dedo indice", True, (200, 200, 200))
    inst_rect = inst_text.get_rect(center=(WINDOW_WIDTH // 2, header_height // 2))
    surface.blit(inst_text, inst_rect)
    
    # Información derecha: Score y Longitud
    score_text = small_font.render(f"Score: {score}", True, TEXT_COLOR)
    score_rect = score_text.get_rect()
    score_rect.topright = (WINDOW_WIDTH - 10, 5)
    surface.blit(score_text, score_rect)
    
    length_text = small_font.render(f"Longitud: {snake_length}", True, ACCENT_COLOR)
    length_rect = length_text.get_rect()
    length_rect.topright = (WINDOW_WIDTH - 10, 25)
    surface.blit(length_text, length_rect)

def main():
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede abrir la camara")
        print("Verifica que tu camara este conectada y no este siendo usada por otra aplicacion")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    snake = Snake()
    food = Food()
    score = 0
    game_over = False
    prev_index_pos = None
    direction_history = []  # Historial para suavizado
    direction_text = "Esperando..."
    
    # Asegurar que la comida no esté en la serpiente
    food.position = food.generate_position(snake.body)
    
    frame_counter = 0
    
    print("Iniciando Snake con Deteccion de Dedo...")
    print("Asegurate de que tu camara este encendida")
    print("Mueve tu dedo indice para controlar la serpiente")
    
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
                    direction_history = []  # Resetear historial al reiniciar
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Leer frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("Advertencia: No se puede leer la camara. Verifica que este conectada.")
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
            direction, prev_index_pos, direction_history = detect_direction(
                hand_landmarks, prev_index_pos, direction_history
            )
            
            if direction:
                snake.change_direction(direction)
                # Actualizar texto de dirección
                dir_map = {
                    (1, 0): "Derecha",
                    (-1, 0): "Izquierda",
                    (0, 1): "Abajo",
                    (0, -1): "Arriba"
                }
                direction_text = dir_map.get(direction, "Desconocido")
        else:
            prev_index_pos = None
            direction_history = []  # Resetear historial cuando no hay mano
            direction_text = "Esperando mano..."
        
        # Mostrar texto en la cámara
        cv2.putText(frame, f"Direccion: {direction_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Mueve tu dedo indice", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar ventana de la cámara
        cv2.imshow("Camara - Mueve tu dedo indice", frame)
        
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
        
        # Dibujar grid solo en el área de juego (no en el header)
        header_height = HEADER_HEIGHT * GRID_SIZE
        game_area = pygame.Rect(0, header_height, WINDOW_WIDTH, WINDOW_HEIGHT - header_height)
        pygame.draw.rect(screen, BACKGROUND, game_area)
        
        # Dibujar grid en área de juego
        for x in range(0, WINDOW_WIDTH, GRID_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (x, header_height), (x, WINDOW_HEIGHT))
        for y in range(header_height, WINDOW_HEIGHT, GRID_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (WINDOW_WIDTH, y))
        
        if not game_over:
            food.draw(screen)
            snake.draw(screen)
        else:
            # Pantalla de game over
            game_over_text = font.render("GAME OVER", True, (255, 50, 50))
            score_text = font.render(f"Puntuacion: {score}", True, TEXT_COLOR)
            restart_text = small_font.render("Presiona R para reiniciar", True, ACCENT_COLOR)
            
            text_rect = game_over_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
            score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
            
            screen.blit(game_over_text, text_rect)
            screen.blit(score_text, score_rect)
            screen.blit(restart_text, restart_rect)
        
        # Dibujar header con toda la información
        fps = clock.get_fps()
        draw_header(screen, direction_text, fps, score, len(snake.body))
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS para suavidad visual
    
    # Limpiar
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

