Ку# DoomWorm

## Цель проекта

**DoomWorm** — экспериментальный проект, в котором connectome **C. elegans** используется как управляющий контроллер.

Конечная цель:

```text
C. elegans brain
      ↓
получает сенсорные сигналы из Doom
      ↓
обрабатывает их через сеть нейронов
      ↓
управляет персонажем
      ↓
учится двигаться, искать цель, избегать опасности и сражаться
```

Проект НЕ должен начинаться сразу с Doom.

Нужно двигаться строго от простого к сложному.

Главный принцип:

> Каждый следующий этап добавляет только одну новую сложность.

---

# 1. Основная гипотеза

Проверить:

> Может ли реальная топология нервной системы C. elegans использоваться как обучаемый универсальный контроллер поведения в среде, отличной от естественной среды червя?

Нас интересует не качество игры в Doom само по себе.

Нас интересуют:

- способность connectome генерировать поведение;
- способность обучаться;
- способность переносить поведение между средами;
- роль реальной биологической топологии;
- сравнение с сетью такого же размера со случайными связями.

---

# 2. Жёсткие ограничения проекта

## 2.1. Не использовать большую нейросеть вместо мозга

Запрещено:

```text
Doom
 ↓
CNN
 ↓
Transformer
 ↓
C. elegans
 ↓
action
```

В таком случае играет CNN/Transformer, а C. elegans становится декоративным слоем.

Допустимо:

```text
environment
 ↓
простое преобразование сенсоров
 ↓
C. elegans
 ↓
простое преобразование motor output
 ↓
environment
```

---

## 2.2. Connectome должен оставаться отдельным слоем

Архитектура:

```text
Environment
    ↓
Sensory Adapter
    ↓
Brain Simulator
    ↓
Motor Adapter
    ↓
Environment
```

Компоненты должны быть независимыми.

Это позволит заменить:

```text
2D world
    ↓
Doom
```

не переписывая мозг.

---

## 2.3. Не начинать с полной биофизической модели

Первая версия нейрона:

```text
Leaky Integrate-and-Fire
```

или ещё более простая дискретная модель.

Не начинать с:

- Hodgkin-Huxley;
- NEURON;
- детальной модели тела;
- мышечных клеток;
- fluid simulation.

Это отдельные будущие этапы.

---

# 3. Общая архитектура

```text
+----------------------+
|     ENVIRONMENT      |
|                      |
|  2D / Doom / other   |
+----------+-----------+
           |
           | observations
           v
+----------------------+
|   SENSORY ADAPTER    |
+----------+-----------+
           |
           | neural stimulation
           v
+----------------------+
|                      |
|     C. ELEGANS       |
|                      |
|     302 neurons      |
|                      |
+----------+-----------+
           |
           | neural activity
           v
+----------------------+
|    MOTOR ADAPTER     |
+----------+-----------+
           |
           | actions
           v
+----------------------+
|     ENVIRONMENT      |
+----------------------+
```

---

# 4. Рекомендуемый стек

Первая версия:

```text
Python 3.12+
NumPy
NetworkX
Matplotlib
pytest
```

Позже:

```text
Godot
или
простая Python 2D-среда
```

Для Doom:

```text
ViZDoom
Gymnasium
```

Для connectome:

```text
OpenWorm / cect
```

Но проект не должен жёстко зависеть от конкретного connectome loader.

Создать собственный внутренний формат графа.

---

# 5. Структура проекта

```text
doomworm/
|
|-- README.md
|
|-- brain/
|   |-- neuron.py
|   |-- synapse.py
|   |-- network.py
|   |-- simulator.py
|
|-- connectome/
|   |-- loader.py
|   |-- model.py
|   |-- mappings.py
|
|-- environments/
|   |
|   |-- simple_2d/
|   |
|   |-- maze/
|   |
|   |-- doom/
|
|-- adapters/
|   |-- sensory.py
|   |-- motor.py
|
|-- learning/
|   |-- fitness.py
|   |-- evolution.py
|   |-- plasticity.py
|
|-- experiments/
|   |-- baseline.py
|   |-- random_network.py
|   |-- real_connectome.py
|
|-- visualization/
|   |-- brain.py
|   |-- metrics.py
|
|-- tests/
|
+-- docs/
```

---

# 6. Этап 0. Самая простая нейронная сеть

## Цель

Не использовать пока C. elegans.

Нужно проверить сам симулятор.

Сделать:

```text
Input neuron
      ↓
Hidden neuron
      ↓
Output neuron
```

Пример:

```text
[INPUT] ---> [N1] ---> [OUTPUT]
```

Input получает:

```text
0.0 .. 1.0
```

Output должен изменять своё состояние.

## Нужно реализовать

```text
Neuron
Synapse
Network
Simulator.step()
```

У каждого нейрона минимум:

```text
id
potential
threshold
decay
activity
```

У synapse:

```text
source
target
weight
```

---

## Acceptance Criteria

Этап считается завершённым, если:

```text
input = 0
→ output не активен

input = 1
→ сигнал проходит по сети
→ output активируется
```

Должны быть unit tests.

---

# 7. Этап 1. Маленький искусственный организм

## Цель

Создать минимального агента.

Пока без C. elegans.

Мир:

```text
+----------------------+
|                      |
|          O           |
|                      |
|     AGENT            |
|                      |
+----------------------+
```

Где:

```text
O = obstacle
```

Агент имеет:

```text
sensor_left
sensor_front
sensor_right
```

Выходы:

```text
motor_left
motor_right
```

---

## Движение

Использовать differential drive:

```text
left motor = 1
right motor = 1
→ forward


left motor = 0
right motor = 1
→ turn left


left motor = 1
right motor = 0
→ turn right
```

---

## Пока без обучения

Сеть можно настроить руками.

Например:

```text
front obstacle
      ↓
right motor increases
left motor decreases
      ↓
turn
```

---

## Acceptance Criteria

Агент должен:

```text
ехать
+
реагировать на сенсоры
+
поворачивать
```

Не обязательно хорошо объезжать препятствия.

---

# 8. Этап 2. Мотивация

## Цель

Агент должен получить причину двигаться.

Добавить:

```text
food
```

Мир:

```text
+----------------------+
|                 FOOD |
|                      |
|      ####            |
|                      |
| AGENT                |
+----------------------+
```

---

## Сенсоры еды

Минимум:

```text
food_left
food_front
food_right
```

Сигнал можно вычислять как:

```text
signal = 1 / distance
```

с ограничением:

```text
0.0 .. 1.0
```

---

## Внутреннее состояние

Добавить:

```text
hunger
```

Пример:

```text
0.0 = сыт
1.0 = сильный голод
```

С каждым tick:

```text
hunger += small_value
```

При достижении еды:

```text
hunger = 0
```

---

# 9. Этап 3. Reward

Добавить reward.

Не награждать непосредственно за движение.

Плохо:

```text
move forward
→ +1
```

Хорошо:

```text
нашёл еду
→ +10

столкнулся
→ -5

умер от голода
→ -20
```

Главный принцип:

```text
movement
не является целью

movement
является способом удовлетворить потребность
```

---

# 10. Этап 4. Обучение маленькой сети

Не использовать пока C. elegans.

Сначала убедиться, что обучение вообще работает.

Рекомендуемая первая стратегия:

```text
Evolutionary Algorithm
```

---

## Алгоритм

```text
создать 100 brains
       ↓
запустить каждого в мире
       ↓
посчитать fitness
       ↓
выбрать лучших
       ↓
копировать
       ↓
случайно изменить веса
       ↓
следующее поколение
```

---

## Fitness

Пример:

```text
+100  food reached

+10   новая область карты

-50   collision

-100  starvation
```

Не переусложнять.

---

## Acceptance Criteria

Через N поколений агент должен:

```text
двигаться к еде
+
реже врезаться
```

---

# 11. Этап 5. Добавить настоящий C. elegans connectome

Только теперь подключить настоящий connectome.

Цель:

```text
302 neurons
```

Загрузить published connectome dataset.

Нужно преобразовать его во внутренний формат:

```text
Neuron[]
Synapse[]
```

Не позволять остальному проекту зависеть от формата исходного dataset.

---

## Внутренний формат

Пример:

```text
Neuron:
    id
    name
    type
    metadata
```

```text
Connection:
    source
    target
    weight
    connection_type
```

Тип:

```text
CHEMICAL
ELECTRICAL
```

---

# 12. Этап 6. Проверить connectome без мира

Перед подключением агента проверить сеть отдельно.

Нужно уметь:

```text
stimulate("ASHL")
```

и увидеть распространение активности.

Визуализация:

```text
ASHL *
     |
     v
    AVA *
   /   \
  *     *
```

---

## Acceptance Criteria

Можно:

```text
выбрать neuron
↓
подать stimulus
↓
увидеть активность downstream neurons
```

---

# 13. Этап 7. Sensory mapping

Теперь связать мир с реальными sensory neurons.

Для начала использовать биологически разумное соответствие.

Пример:

```text
food signal
     ↓
chemosensory neurons


obstacle/touch
     ↓
mechanosensory neurons


danger
     ↓
nociceptive neurons
```

Не требуется идеальная биологическая точность.

Но mapping должен быть:

```text
явным
конфигурируемым
документированным
```

---

# 14. Этап 8. Motor mapping

Нужно превратить активность motor/interneurons в движение.

Пример:

```text
forward-related activity
           ↓
 left wheel + right wheel


turn/reversal activity
           ↓
 differential wheel speed
```

Motor Adapter должен быть отдельным компонентом.

Не записывать игровую логику внутрь мозга.

---

# 15. Этап 9. Первый DoomWorm

Теперь получаем:

```text
2D world
    ↓
sensory adapter
    ↓
C. elegans
302 neurons
    ↓
motor adapter
    ↓
2D agent
```

Пока без обучения.

Посмотреть, появляется ли вообще осмысленное поведение.

Логировать:

```text
position
heading
sensor values
active neurons
motor output
collisions
food reached
```

---

# 16. Этап 10. Обучаем C. elegans

Структура connectome фиксирована.

Запрещено сначала изменять topology.

Обучаем:

```text
synaptic weights
```

Первая стратегия:

```text
evolution
```

Позже:

```text
reward-modulated STDP
```

---

## Ограничение

```text
connectome topology = FIXED
```

Разрешено:

```text
weights = trainable
```

---

# 17. Этап 11. Сравнение

Создать минимум три варианта.

## A. Real Worm

```text
302 neurons
real topology
```

## B. Random Worm

```text
302 neurons
same approximate connection count
random topology
```

## C. Shuffled Worm

```text
302 neurons
degree distribution approximately preserved
edges shuffled
```

---

## Сравнивать

```text
food reached
collisions
survival time
distance travelled
generalization
learning speed
```

Это обязательная часть проекта.

Иначе невозможно понять, даёт ли настоящий connectome какое-либо преимущество.

---

# 18. Этап 12. Новые карты

Не обучать на одной комнате.

Генерировать:

```text
random start
random food
random obstacles
```

Пример:

```text
MAP A

################
# A       #   F#
#         #    #
################


MAP B

################
# F             #
#      ####     #
#             A #
################
```

---

## Acceptance Criteria

Обученный мозг должен работать на карте, которую раньше не видел.

---

# 19. Этап 13. Убрать еду и добавить цель

Перейти от биологической задачи:

```text
find food
```

к абстрактной:

```text
reach target
```

Target становится новым привлекательным сигналом.

Это первый шаг к Doom.

---

# 20. Этап 14. Добавить опасность

Добавить объект:

```text
ENEMY
```

Пока enemy не двигается.

Он создаёт:

```text
danger signal
```

Если агент касается enemy:

```text
damage
```

Задача:

```text
TARGET attraction
+
ENEMY avoidance
```

---

# 21. Этап 15. Мини-Doom без Doom

Создать очень простой 2D сценарий:

```text
########################
#                      #
#     ENEMY            #
#                      #
#            ####      #
#                      #
# AGENT           EXIT #
########################
```

Управление:

```text
FORWARD
TURN_LEFT
TURN_RIGHT
```

Цель:

```text
reach EXIT
avoid ENEMY
```

---

# 22. Этап 16. Добавить FIRE

Теперь появляется четвёртое действие:

```text
FIRE
```

Нужно выбрать neural activity pattern, который Motor Adapter преобразует в:

```text
FIRE
```

Это искусственный motor mapping.

Это нормально.

Главное:

```text
brain topology
не менять специально под FIRE
```

---

# 23. Этап 17. Мини-Doom combat

Сценарий:

```text
AGENT
   |
   |
   v

        ENEMY

        EXIT
```

Reward:

```text
enemy hit
→ positive

enemy killed
→ larger positive

damage received
→ negative

death
→ large negative

exit reached
→ positive
```

---

# 24. Этап 18. Подключить ViZDoom

Только теперь добавить настоящий Doom.

Первый сценарий должен быть очень простой.

Не использовать обычную карту Doom.

Создать custom scenario:

```text
room
+
one player
+
one target
+
no enemy
```

---

# 25. Doom observation V1

Не использовать framebuffer.

Сначала подавать структурированные данные.

Например:

```text
target_left
target_center
target_right

wall_left
wall_center
wall_right

health
```

Получаем:

```text
ViZDoom
   ↓
Game Variables
   ↓
Sensory Adapter
   ↓
C. elegans
```

---

# 26. Doom action V1

Минимум:

```text
MOVE_FORWARD
TURN_LEFT
TURN_RIGHT
```

Не добавлять:

```text
strafe
jump
weapon select
use
```

---

# 27. Doom Level 1

Цель:

```text
START
  ↓
найти EXIT
```

Без врагов.

---

# 28. Doom Level 2

Добавить:

```text
walls
corridors
```

Цель:

```text
navigation
```

---

# 29. Doom Level 3

Добавить:

```text
one stationary enemy
```

Но пока:

```text
avoid enemy
```

Не стрелять.

---

# 30. Doom Level 4

Добавить:

```text
FIRE
```

Enemy:

```text
stationary
```

Цель:

```text
aim approximately
+
shoot
```

---

# 31. Doom Level 5

Enemy начинает двигаться.

---

# 32. Doom Level 6

Несколько комнат.

Теперь одновременно:

```text
navigation
+
danger avoidance
+
combat
```

---

# 33. Doom Level 7

Только теперь попробовать стандартный Doom scenario.

Не ожидать хорошего результата.

На этом этапе задача:

```text
survive
move
avoid walls
react to enemies
occasionally attack
```

---

# 34. Только после этого — изображение

До этого момента DoomWorm не должен получать framebuffer.

Следующий исследовательский этап:

```text
Doom framebuffer
      ↓
very small visual encoder
      ↓
sensory channels
      ↓
C. elegans
```

Но encoder НЕ должен превращаться в полноценную нейросеть, которая самостоятельно решает задачу.

---

# 35. Visual V1

Самый простой вариант:

Разделить экран:

```text
+---------+---------+---------+
|  LEFT   | CENTER  | RIGHT   |
|         |         |         |
+---------+---------+---------+
```

Извлекать:

```text
brightness
motion
enemy presence
obstacle proximity
```

И подавать это как sensory channels.

---

# 36. Visual V2

Позже:

```text
8x8
```

или:

```text
16x16
```

grayscale.

Но не начинать отсюда.

---

# 37. Обучение

Рекомендуемый порядок:

```text
1. Evolution
2. Reward-modulated plasticity
3. STDP
4. combinations
```

Не начинать с reinforcement learning framework.

---

# 38. Режимы эксперимента

Реализовать четыре режима.

## Frozen

```text
topology = fixed
weights = fixed
```

---

## Plastic

```text
topology = fixed
weights = trainable
```

---

## Evolved

```text
topology = fixed
weights = evolutionary optimized
```

---

## Free

```text
topology = trainable
weights = trainable
```

Free нужен только как upper baseline.

---

# 39. Метрики

Обязательно сохранять:

```text
episode
generation
fitness
survival_time
distance
collisions
targets_reached
damage_received
enemies_hit
enemies_killed
```

Дополнительно:

```text
active_neurons
average_activity
synaptic_weight_distribution
```

---

# 40. Replay

Каждый эксперимент должен иметь возможность:

```text
save seed
save brain
save weights
save environment
```

и затем:

```text
replay
```

Результат должен быть воспроизводимым.

---

# 41. Визуализация

Нужен простой debug screen.

```text
+----------------------+----------------------+
|                      |                      |
|      ENVIRONMENT     |      CONNECTOME      |
|                      |                      |
|      agent --->      |  *---*---*           |
|                      |    \ | /             |
|      obstacle        |      *               |
|                      |                      |
+----------------------+----------------------+

Sensors:
front: 0.72
left:  0.15
right: 0.31

Motors:
left:  0.84
right: 0.41

Reward:
+3.14
```

---

# 42. Не оптимизировать раньше времени

Для 302 нейронов достаточно CPU.

Не использовать сначала:

```text
CUDA
GPU
distributed computing
Jetson
Rust optimization
```

Сначала должна работать гипотеза.

---

# 43. Тесты

На каждом уровне нужны тесты.

Минимум:

```text
test_neuron_activation
test_synapse_signal
test_network_step
test_connectome_loader
test_sensor_mapping
test_motor_mapping
test_episode_replay
```

---

# 44. Порядок реализации

Агент обязан идти именно в этом порядке:

```text
[0]
3 neuron network
   ↓
[1]
2D agent
   ↓
[2]
food + hunger
   ↓
[3]
reward
   ↓
[4]
evolution
   ↓
[5]
load C. elegans
   ↓
[6]
stimulate connectome
   ↓
[7]
connect sensors
   ↓
[8]
connect motors
   ↓
[9]
C. elegans controls 2D agent
   ↓
[10]
train weights
   ↓
[11]
real vs random topology
   ↓
[12]
random maps
   ↓
[13]
target
   ↓
[14]
danger
   ↓
[15]
mini Doom
   ↓
[16]
fire
   ↓
[17]
combat
   ↓
[18]
ViZDoom
   ↓
[19]
simple Doom navigation
   ↓
[20]
enemy
   ↓
[21]
combat
   ↓
[22]
real Doom scenario
   ↓
[23]
visual input
```

---

# 45. Главное правило разработки

Нельзя реализовывать следующий этап, пока предыдущий:

```text
не работает
+
не покрыт тестами
+
не имеет observable demo
```

---

# 46. Что считается MVP

DoomWorm MVP — это НЕ Doom.

MVP:

```text
real C. elegans connectome
        ↓
302 simulated neurons
        ↓
3 obstacle sensors
+
3 food sensors
        ↓
2 motor outputs
        ↓
2D differential-drive agent
        ↓
ищет еду
+
обходит часть препятствий
+
может обучаться
```

После этого проект доказал, что базовый pipeline работает.

---

# 47. Что считается первой крупной вехой

```text
DoomWorm
+
ViZDoom
+
MOVE_FORWARD
+
TURN_LEFT
+
TURN_RIGHT
+
custom empty level
+
EXIT
```

C. elegans должен самостоятельно научиться достигать EXIT.

---

# 48. Что считается основной целью проекта

```text
ViZDoom
+
navigation
+
one or more enemies
+
FIRE
+
C. elegans connectome
+
trained synaptic weights
```

DoomWorm должен:

```text
двигаться
не застревать постоянно в стенах
реагировать на врага
стрелять
иногда уничтожать врага
продолжать исследовать уровень
```

Не требуется проходить Doom идеально.

---

# 49. Исследовательская часть

После получения работающей системы провести сравнение:

```text
REAL CONNECTOME
       vs
RANDOM CONNECTOME
       vs
SHUFFLED CONNECTOME
       vs
SMALL ARTIFICIAL RNN/SNN
```

Использовать:

```text
same neuron count
similar connection budget
same environment
same training budget
```

---

# 50. Главный результат

Ответить на вопрос:

> Даёт ли биологическая топология C. elegans полезный inductive bias для обучения поведения в совершенно чужой среде?

Возможны три одинаково полезных результата.

```text
Real > Random
```

Биологическая топология даёт преимущество.

```text
Real ~= Random
```

Топология сильно специализирована под тело и естественную среду.

```text
Real < Artificial
```

Искусственная архитектура лучше для новой задачи.

Все три результата полезны.

---

# 51. Правила для агента

Агент должен:

1. Делать минимальную реализацию каждого этапа.
2. Не усложнять архитектуру без необходимости.
3. Сначала писать тест.
4. После этапа создавать маленький runnable demo.
5. Сохранять результаты экспериментов.
6. Использовать deterministic seed, где возможно.
7. Документировать assumptions.
8. Не заменять C. elegans скрытой ML-моделью.
9. Не переходить к Doom до рабочего 2D агента.
10. Не переходить к framebuffer до рабочего structured-input Doom.
11. При спорном выборе предпочитать более простое решение.
12. Каждую новую функцию добавлять отдельным этапом.
13. Не переписывать рабочие слои без объективной необходимости.

---

# 52. Definition of Done

Проект можно считать успешным, когда существует воспроизводимый pipeline:

```text
C. elegans connectome
        ↓
brain simulator
        ↓
sensory adapter
        ↓
ViZDoom
        ↓
motor adapter
        ↓
actions
        ↓
reward / learning
```

и можно запустить:

```bash
doomworm train
```

а затем:

```bash
doomworm play --brain trained-brain.json
```

и увидеть, как один и тот же сохранённый мозг самостоятельно управляет персонажем Doom.

---

# 53. Первая задача агенту

Не начинать реализацию ViZDoom.

Сделать только:

```text
Neuron
Synapse
Network
Simulator
```

и demo:

```text
3 neurons
```

После этого:

```text
pytest
```

должен проходить полностью.

Затем реализовать минимального 2D-агента с:

```text
3 distance sensors
2 motors
1 obstacle
```

Только после демонстрации работающего движения переходить дальше.
