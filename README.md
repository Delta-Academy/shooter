## RL to play Space Shooter :gun::rocket:

You control a space ship. A rival space-mob has entered your gangs patch.

You must terminate it. And you must terminate it using RL.

![Space shooter starting position](images/space-shooter.png)

## Rules of Space Shooter

Space shooter is a **two-player** top down shooter game. Players take turns synchronously.

The arena is a 2d grid with **4 barriers**.

The game starts with the **two players** positioned behind a randomly chosen barrier in the arena.

Each player has 4 actions: rotate clockwise, rotate anti-clockwise, go forward (in the direction you're facing), and shoot.

At any time, only two bullets from a given player can be present in the arena. If the player shoots twice in quick succession, the player must wait until the bullet goes off the edge of the arena or strikes a barrier before they can shoot again.

In the arena, there are barriers that the players can hide behind. These barriers prevent both players and bullets travelling through them.

You ship is not able to leave the arena, and you'd never do that to your space-gang pals anyway!

**The goal is to shoot the other player before they shoot you**

# Competition Rules :crossed_swords:

1. Your task is to build a **Reinforcement Learning agent** that plays **Space Shooter**.
   - You can only store data to be used in a competition in a neural network (saved in a `.pt` file by `save_network()`)
   - In the competition, your agent will call the `choose_move()` function in `main.py` to select a move (`choose_move()` may call other functions in `main.py`)
   - Any code not in `main.py` will not be used.
   - We **strongly suggest** you use a neural network.
2. Submission deadline: **5pm GMT, Sunday**.
   - You can update your code after submitting, but **not after the deadline**.
   - Check your submission is valid with `check_submission()`
3. The competition is a knockout tournament where your AI will play other teams' AIs 1-v-1
   - Each 1-v-1 matchup consists of a **best of 5 game** with the **starting positions of the players** chosen randomly.

The competition & discussion will be in [Gather Town](https://app.gather.town/app/nJwquzJjD4TLKcTy/Delta%20Academy) at **6pm GMT on Sunday** (60 mins after submission deadline)!

![Ex](images/tournament_tree.png)

## Technical Details :hammer:

### States :space_invader:

The state space is repesented as a **1d numpy array** of length **18**. Each element contains information about the location of objects in the arena.

#### Each object has three elements in the array:

- 1: The x position of the object in the arena
- 2: The y position of the object in the arena
- 3: The angle of the object in the arena (expressed as degrees from north)

#### The following objects are represented in the array (in order):

- The player you control
- Your opponent
- Your first bullet
- Your second bullet
- Your opponents first bullet
- Your opponents second bullet

If you or your opponent have bullets that have not yet been fired, their location is represented by **[0,0,0]**

### Actions :axe:

**Actions** in space shooter are integers from the set {0,1,2,3}. They correspond to the following for your spaceship...

- **0**: Rotate your ship 15 degrees clockwise
- **1**: Rotate your ship 15 degrees anti-clockwise
- **2**: Move forward the direction you're facing
- **3**: Shoot!

### Rewards :moneybag:

You receive `+1` for winning, `-1` for losing and `0` for a draw. You receive `0` for all other moves.

## Functions you write :point_left:

<details>
<summary><code style="white-space:nowrap;">  train()</code></summary>
Write this to train your network from experience in the environment.
<br />
<br />
Return the trained network so it can be saved.
</details>
<details>
<summary><code style="white-space:nowrap;">  choose_move()</code></summary>
This acts greedily given the state and network.

In the competition, the choose_move() function is called to make your next move. Takes the state as input and outputs an action.

Also has a verbose mode, which when set to True prints to console the legal moves. Useful for debugging.

</details>

## Existing Code :pray:

### Need to Know

<details>
<summary><code style="white-space:nowrap;">  Env</code> class</summary>
The environment class controls the game and runs the opponent. It should be used for training your agent.
<br />
<br />
See example usage in <code style="white-space:nowrap;">play_shooter()</code>.
<br />
<br />
The opponent's <code style="white-space:nowrap;">choose_move</code> function is input at initialisation (when <code style="white-space:nowrap;">Env(opponent_choose_move)</code> is called). The first player is chosen at random when <code style="white-space:nowrap;">Env.reset()</code> is called. Every time you call <code style="white-space:nowrap;">Env.step()</code>, 2 moves are taken - yours and then your opponent's. Your opponent sees a 'flipped' version of the board, where his pieces are shown as <code style="white-space:nowrap;">1</code>'s and yours are shown as <code style="white-space:nowrap;">-1</code>'s.
    <br />
    <br />
    Both <code style="white-space:nowrap;">  Env.step()</code> and <code style="white-space:nowrap;">  Env.reset()</code> have <code style="white-space:nowrap;">  verbose</code> arguments which print debugging info to console when set to <code style="white-space:nowrap;">True</code>. Verbose visualises the ongoing progress of the game in the console. Player1's tiles (you) are represented as an X. Your opponents tiles are represented as O.
</details>

<details>
<summary><code style="white-space:nowrap;">  choose_move()</code></summary>
This acts greedily given the state and value network.
<br />
<br />
In the competition, the <code style="white-space:nowrap;">choose_move()</code> function is called to make your next move. Takes the state as input and outputs an action.
<br />
<br />
Also has a <code style="white-space:nowrap;">verbose</code> mode, which when set to <code style="white-space:nowrap;">True</code> prints to console the possible actions, their corresponding features if taken and the values of those feature vectors. Useful for debugging.
</details>

<details>
<summary><code style="white-space:nowrap;">  choose_move_randomly()</code></summary>
Like above, but randomly picks from available legal moves.
<br />
<br />
Takes the state as input and outputs an action.
</details>

<details>
<summary><code style="white-space:nowrap;">  human_player()</code></summary>
    Use me as a function to play against your both with the keyboard!
    Use the direction keys to move and the spacebar to shoot!
<br />
<br />
Takes the state as input and outputs an action.
</details>

<details>
<summary><code style="white-space:nowrap;">  play_shooter()</code></summary>
Plays 1 game of shooter, which is visualised graphically. (if <code style="white-space:nowrap;">render=True</code>)
<br />
<br />
Inputs:

<code style="white-space:nowrap;">your_choose_move</code>: Function that takes the state and outputs the action for your agent.

<code style="white-space:nowrap;">opponent_choose_move</code>: Function that takes the state and outputs the action for the opponent.

<code style="white-space:nowrap;">game_speed_multiplier</code>: controls the gameplay speed. High numbers mean fast games, low numbers mean slow games.

<code style="white-space:nowrap;">verbose</code>: whether to print to console each move and the corresponding board states.

</details>

## Other code :gear:

There are **a load** of functions in `game_mechanics.py`. The useful functions are clearly indicated and are explained in their docstrings. **Feel free to use these, but don't change them.** This is because the original `game_mechanics.py` file will be used in the competition.

If you want to tweak one, copy-paste it to `main.py` and rename it.

## Suggested Approach :+1:

1. Discuss your neural network architecture - how many inputs, outputs, hidden layers & which activation functions should you use
2. **Write `train()`** (you can borrow code from past exercises).
3. Insert debugging messages - you want to make sure that:
   - Loss is decreasing :chart_with_downwards_trend:
   - The magnitude of update steps are decreasing :arrow_down:
   - Performance on shooter is improving :arrow_up:
4. Iterate on the neural network architecture, hyperparameters & training algorithm
