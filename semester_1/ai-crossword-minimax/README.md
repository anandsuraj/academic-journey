  



# Two-Player Crossword Puzzle Game using Minimax Algorithm

## 🎯 Objective

Develop a **two-player crossword puzzle game** that uses the **Minimax algorithm** to simulate both interactive and AI-based automated play. The game is implemented in **Python** with dynamic board updates, score tracking, and intelligent move decisions.

  

---

## 🧠 Game Features

- 12x9 crossword grid initialized with `#`

- First word ("RABBIT") is always placed at a fixed position

- Turn-based word placement

- Valid word crossing enforced

- Minimax AI computes best scoring moves

- Score updates per valid/invalid placement

- Game ends when no valid moves remain

  

---

  

## 🧮 Scoring Rules

  

- Correct placement: +1 point per letter

- Invalid move: −1 point

- Scoreboard updates after each turn

- Final result shows the best two scoring outcomes (in AI mode)

  

---

  

## 🕹️ Game Modes

  

### 1. Interactive Mode (2 Players)

- Players alternate turns and input word placements.

- Board is updated and displayed after every move.

  

### 2. AI Mode (Automated)

- Minimax algorithm selects optimal moves.

- Game simulates 10 iterations.

- Two best scoring results are displayed.

  

---

  

## ▶️ How to Run

  

```bash
# Clone the repository
git clone https://github.com/anandsuraj/ai-crossword-minimax.git

# Move into the project directory
cd ai-crossword-minimax
  
# Run the game

python  crossword_game.py or .ipynb file

````

  

---

  

## 📸 Sample Output (Interactive Mode)

  

```text

Game started with RABBIT placed at (1,5) direction D

Player 2 placed CAT at (6, 3) direction A

Score: Player 1: 6, Player 2: 3

Player 1 placed CAMEL at (6, 3) direction D

Score: Player 1: 11, Player 2: 3
...

```

  

---

## 🛠️ Tech Stack

  

*  **Language**: Python 3

*  **Algorithm**: Minimax (No alpha-beta pruning)

*  **Input/Output**: Console

---
## 📌 Notes

* No random seed used – results vary across runs.

* Edge cases handled: overlapping, boundary checks, valid crossings.

* Dynamic input-based word list for flexibility.
---
