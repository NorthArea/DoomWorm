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

Активность нейрона:

```text
этапы 0-4   бинарная (0 / 1), порог + сброс
этап 5+     graded, в диапазоне [0, 1]
            например activity = clip(potential / threshold, 0, 1)
            без сброса потенциала
```

Причина: бинарная сеть не умеет сравнивать силу сигналов
(два сработавших датчика гасят оба мотора), а веса коннектома —
это счётчики синапсов, при бинарных спайках сеть либо взрывается,
либо молчит. Порог и утечка сохраняются, модель остаётся LIF-подобной.

Веса при загрузке коннектома нормируются.

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

## 3.1. Временные масштабы

Сигнал проходит один синапс за тик мозга. В коннектоме путь от сенсора
до мотора занимает 3-5 хопов, поэтому один шаг среды должен содержать
несколько тиков мозга:

```text
brain_steps_per_env_step = 5..10   (параметр, по умолчанию 5)
```

Motor Adapter усредняет активность моторных нейронов по этому окну.
Этапы 0-4 могут использовать 1:1.

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

2D-среда пишется на Python внутри проекта (никаких игровых движков).

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

## Конец эпизода

Эпизод заканчивается, когда выполняется любое из:

```text
hunger >= 1.0      (смерть от голода)
tick >= max_steps  (лимит)
```

`hunger_rate` подбирать так, чтобы без еды агент жил 300-500 тиков
(на карте 20x20 при скорости 0.2 это 0.002-0.005 за тик).

---

## Респаун еды

Съеденная еда появляется снова в случайной точке карты.
Случайность берётся из seed эпизода, чтобы прогон был воспроизводим.

Респаун нужен уже на этапе 4: иначе эволюция выучит одну карту.

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

столкновение (за каждый тик контакта)
→ -0.5

новая клетка карты 1x1
→ +0.1

умер от голода
→ -20
```

Столкновение штрафуется за тик, а не за событие: агент, упёршийся
в стену, должен хотеть оттуда уехать. Штраф маленький, чтобы не
перевесить еду. Суммарный штраф за столкновения в эпизоде ограничен.

Reward — единственный источник чисел. Fitness на этапе 4 — это сумма
reward за эпизод, отдельной таблицы fitness нет.

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

```text
fitness = сумма reward за эпизод (см. этап 3)
```

Оценивать каждый мозг на нескольких seed-картах (случайный старт,
еда с респауном) и усреднять. Одна карта = оверфит.

Не переусложнять.

---

## Acceptance Criteria

Через N поколений агент должен:

```text
двигаться к еде
+
реже врезаться
```

На этом этапе появляется формат сохранения мозга (см. §40 Replay):
лучший мозг поколения сохраняется в JSON и может быть проигран заново.

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

## Датасет

Первый источник: Cook et al. 2019 (wormwiring.org), гермафродит,
химические синапсы + gap junctions, 302 нейрона.
CSV кладётся в `data/connectome/` вместе с файлом лицензии и ссылкой.
Альтернативы на потом: Witvliet 2021, White 1986 через OpenWorm.

---

## Знаки синапсов

Датасет даёт количество синапсов, не знак.

```text
инициализация:
    GABA-нейроны (по данным о нейромедиаторах)  → тормозные, вес < 0
    остальные                                   → возбуждающие, вес > 0
    величина ∝ числу синапсов, нормирована

обучение:
    вес может менять знак
```

---

## Электрические синапсы

В v1 gap junction = пара симметричных возбуждающих связей A→B и B→A.
Честная модель на разности потенциалов — отдельный будущий этап.

---

## Никаких добавленных нейронов

С этапа 5 в сети только 302 нейрона коннектома. Искусственные
интернейроны (как HUNGRY на этапе 2) запрещены — это изменение
топологии. Внутренние состояния (голод и т.п.) подаются как ток
на конкретные биологические нейроны, гейтинг делает сама сеть
или Sensory Adapter.

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

На этом этапе делается debug screen из §41 (сохранённые PNG/GIF).

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

Стартовое соответствие (менять можно, но фиксировать в конфиге):

```text
obstacle / touch     → ALM, AVM, PLM, FLP   (mechanosensory)
danger / nose touch  → ASH                  (nociceptive)
food                 → AWA, AWC, ASE        (chemosensory)
hunger               → NSM, ASI             (внутреннее состояние)
```

Лево/право: у червя это не L/R-пары нейронов, а дорсо-вентральные
взмахи головой. Наше отображение сенсоров left/right на L/R-нейроны
заведомо искусственное. Это допустимо, но записывается как допущение.
Сенсор front подаётся на оба нейрона пары.

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

Стартовое соответствие:

```text
forward   ← AVB, PVC, B-класс мотонейронов
reversal  ← AVA, AVD, A-класс мотонейронов
turn      ← SMD, RIV (асимметрия D/V → лево/право)
```

Активность усредняется по окну `brain_steps_per_env_step`
и отображается в непрерывные скорости колёс [0, 1].

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

Перед началом этапа проверить, что `vizdoom` ставится колесом
под текущую платформу (macOS arm64) и версию Python:

```bash
uv pip download vizdoom
```

Сборка из исходников (cmake, SDL2, boost) не вариант: глобальные
установки и brew запрещены. Если колеса нет — отдельный venv
на Python 3.12 только для Doom-этапов. Проверять заранее, не в
момент начала этапа.

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

# 27. Этап 19. Doom Level 1

Цель:

```text
START
  ↓
найти EXIT
```

Без врагов.

---

# 28. Этап 20. Doom Level 2

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

# 29. Этап 21. Doom Level 3

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

# 30. Этап 22. Doom Level 4

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

# 31. Этап 23. Doom Level 5

Enemy начинает двигаться.

---

# 32. Этап 24. Doom Level 6

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

# 33. Этап 25. Doom Level 7

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

# 34. Этап 26. Только после этого — изображение

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

Формат: один JSON-файл на мозг.

```text
{
  "format": 1,
  "neurons":  [{id, threshold, decay, ...}],
  "synapses": [{source, target, weight, type}],
  "meta":     {seed, stage, generation, fitness, world_config}
}
```

Появляется на этапе 4, дальше не меняется без bump «format».

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

Реализуется на этапе 6 как сохранённые кадры (PNG) и GIF из
matplotlib. Живое окно не нужно: анимация в matplotlib медленная,
а игровых библиотек в стеке нет.

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
Doom L1: пустая комната, EXIT
   ↓
[20]
Doom L2: стены, коридоры
   ↓
[21]
Doom L3: неподвижный враг, избегать
   ↓
[22]
Doom L4: FIRE по неподвижному врагу
   ↓
[23]
Doom L5: враг двигается
   ↓
[24]
Doom L6: несколько комнат
   ↓
[25]
Doom L7: стандартный сценарий
   ↓
[26]
visual input
```

Номера этапов здесь совпадают с заголовками разделов выше.

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
FREE (режим Free из §38: тот же симулятор, свободная топология)
```

Четвёртый вариант — не отдельная RNN на градиентах, а тот же
симулятор с обучаемой топологией. Другой стек обучения сделал бы
сравнение нечестным.

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

---

# 54. История изменений плана

## 2026-09-14 — ревью после этапов 0-2

- §2.3: graded-активность [0, 1] с этапа 5; бинарная только на этапах 0-4.
- §3.1: несколько тиков мозга на шаг среды, усреднение в Motor Adapter.
- §4: убран Godot.
- §8: конец эпизода (голод / лимит), диапазон `hunger_rate`, респаун еды с этапа 4.
- §9-10: единая шкала reward, fitness = сумма reward, штраф за столкновение за тик,
  оценка на нескольких seed-картах.
- §11: датасет Cook 2019, знаки синапсов из нейромедиаторов, gap junction как
  симметричная пара, запрет искусственных нейронов с этапа 5.
- §12, §41: debug screen делается на этапе 6, как PNG/GIF.
- §13-14: стартовые таблицы sensory/motor mapping, L/R — записанное допущение.
- §24: проверка колеса vizdoom до начала этапа 18, запасной venv на 3.12.
- §40: JSON-формат мозга, появляется на этапе 4.
- §27-34, §44: Doom-уровни пронумерованы как этапы 19-26, список §44 совпадает
  с заголовками.
- §49: четвёртый бейзлайн — режим Free, не отдельная RNN.
