# Virtual Reality Room

- VR version of 'escape room' game

- Basic Rules

  - Escape a room in a given time
  - Room is consisted of several objects. Some can be touched and moved, but some can't.
  - Also, some objects contains a hint, and some don’t. 
  - Based on the hints you found, you should find the way to solve the quiz which can help you escape the room.
  
- Concept of Game

  - The player has a special ability called psychometry (an ability to read objects' memory or vision)
    - If you select objects, they will give you the hint to solve the quizzes
    - Some objects contain direct hints or extra problems to solve
    - Once you solve all the problems a stair will pop up. This means you have cleared the game!

- Room Design

  - Size : X = (-10, 10), Y = (-10, 10), Z = (-10, 20)
  - Only one player(person) is in the room
  - Starting Eye Position : (0, -3, 12)
  - If a player sits on the floor, the eye position changes to (0, -6, 12)
  - A player can move forward, backward, left, and right. Also, the view direction can be changed up, down, left, and right
  - There is restriction on moving area and view direction for a realistic effect
  - View range : X = (-2, 2), Y = (-2, 2), Z = (-3, -40)
  - Objects : Table, sofa, fireplace, bunny, clocks, ball, cubes and stairs
  
- How to solve it [Hints]

  - Some objects contains hints
  - It will give you the image if you select it
  - The cubes with the numbers are the direct hint
  - You need to search all the cubes in this room first to solve the question
  - These cubes are movable, scalable, and rotatable
  - If you see the ceiling there are 7 fireballs
  - If you click it, its color will change. These are the main problems of this game.
  - If you successfully match all the colors, the stairs will come out

- Textures & Lighting

  - There are 9 textures. All objects contain only one texture except the fireplace. The fireplace consists of brick, concrete, and wood
  - Light position : (0,0,0)
  - For shading the phong values are added up (power of 32)

- Next Step / Possible Improvement

  - Shadow
  - Collision control
  - More meshes (Add locks on the objects)
 
- Image

<img width="640" alt="Screen Shot 2019-12-14 at 9 16 01 PM" src="https://user-images.githubusercontent.com/23174275/75495314-3e13cd00-598c-11ea-9da9-d238c0c05cbc.png">

