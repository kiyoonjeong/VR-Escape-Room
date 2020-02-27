# Building Virtual Reality Room

- Created a game called 'Escaping room', which is really hot offline game these days.

- Basic Rules

  - Escape room in a given time
  - Room is consisted of several objects. Some could be touched and moved, but some don’t. 
  - Also, some objects contains the hint, and some don’t. 
  - Based on the hints you found, you should find the way to solve the quiz. (Finding quiz is also one of the quiz)
  
- Concept of Game

  - The player has a special ability called psychometry, which enable player to read the objects memory(or vision)
    - If you select some objects, they will give you the hint to solve the questions
    - The other objects would contains direct hints or problems
    - In this game, there is only one problem to be solved. If you solve it, the stair will come out, which means success to escape.

- Room Design

  - Size : X= (-10, 10), Y = (-10, 10), Z = (-10, 20)
  - Only one player(person) is in the room
  - Starting Eye Position : (0, -3, 12)
  - If player sit, the eye position is changed to (0, -6, 12)
  - Player can move forward, backward, left, and right. Also, the view direction can also be changed up, down, left, and right
  - There is restriction on moving area and view direction to make it real
  - View range : X = (-2, 2), Y = (-2, 2), Z= (-3, -40)
  - Objects : Table, sofa, fireplace, bunny, clocks, ball, cubes and stair
  
- How to solve it [Hints]

  - Some objects contains hints
  - It will give you the image if you select it
  - The cubes with the numbers are the direct hint
  - You need to search the whole cubes in this room first to solve the question
  - These cubes are movable, scalable, and rotatable
  - If you see the ceiling, there are 7 fireballs
  - If you click it, its color would be changed. These are the problems to be solved
  - If the answer is correct, the stair will come out

- Textures & Lighting

  - I used 9 textures. All objects contain only one texture except fireplace. It is consisted of brick, concrete, and wood
  - Light position : (0,0,0)
  - For shading, I added up the phong value (power of 32)

- Next Step / Possible Improvement

  - Shadow
  - Collision control
  - More meshes (Add locks on the objects)
 
- Image

