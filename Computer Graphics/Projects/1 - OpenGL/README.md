**"Minecraft 2" - Code**

Yes. The code was compressed all in one file. It was an academic project and it was easier to submit that way. Sorry about that!

[See project page at luvas.io](https://luvas.io/portfolio/minecraft2)

---

**How to run (linux)**

Install openGL/GLUT

``sudo apt-get install freeglut3-dev``

Before compiling, you might wanna change configuration inside "codigo.cpp". Mainly "MOUSE_SENSE".

Compile and run

``g++ codigo.cpp -lGL -lGLU -lglut -o app`` ``./app``

---

**How to play**

# Game Modes

## 1. Spectator (Default)
- **Collision:** Disabled.
- **Gravity:** Disabled.
- **Camera:** Movable relative to the normal (free camera).

## 2. Survival
- **Collision:** Enabled.
- **Gravity:** Enabled.
- **Camera:** Movable relative to the XZ plane.

---

# General Controls

- **Key “G”**: Switches between Spectator and Survival modes.
- **Key “R”**: Regenerates the map from a random “seed”.
- **Key “P”**: Toggles debug mode.
  - Displays the camera hitbox.
  - Shows Bezier curves between the camera and destroyed blocks (block entities).
- **Arrow keys** or **Mouse movement**: Rotate the camera.
- **Mouse “Left”**: Breaks the block the camera is pointing at (indicated by the darker hitbox).
- **Mouse “Right”**: Places a block at the indicated position (shown by the lighter hitbox).
- **Mouse “Scroll”**: Switches the block to be placed using the right-click.
  - (Current block can be checked in the console).

---

# Spectator Mode Controls

- **W, A, S, D**: Move the camera relative to the normal.
- **Space**: Move the camera up (along the Y-axis).
- **Shift**: Move the camera down (along the Y-axis).

---

# Survival Mode Controls

- **W, A, S, D**: Move the camera relative to the XZ plane.
- **Space**: Jump (along the Y-axis).

---

# What Can Be Done?

1. Move freely through the world.
2. Break, place, and collect different types of blocks.
3. Regenerate the map.

---

# Application of Work Requirements

1. **3D Modeling using Primitives**  
   Used to build the visual objects of the simulation (blocks, block entities, clouds, hitboxes, etc.).

2. **Geometric Transformations**  
   Used to determine the position and rotation of the camera, hitboxes, and animation frames of block entities.

3. **Animations / Time Control**  
   Implemented with a timer to determine the position/rotation of block entities and hitboxes at a given moment.

4. **Mouse and Keyboard Control**  
   Used for camera movement, map controls, and game mode switches.

5. **Camera Positioning and Perspective**  
   A free camera model is used, and the projection is perspective.

6. **Lighting**  
   Used for global lighting of blocks, with shading variations based on the observer’s position.

7. **Parametric Curves**  
   Used to calculate the trajectory between a block entity and the camera when close.
   - This is explicitly visible when debug mode is enabled.