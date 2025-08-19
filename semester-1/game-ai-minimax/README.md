# 🎯 Strategic Game AI: Crossword Puzzle with Minimax Algorithm

## 🎮 Project Overview

A sophisticated **two-player crossword puzzle game** implementing the **Minimax algorithm** for intelligent gameplay. This project demonstrates advanced search algorithms, game theory principles, and AI decision-making in a competitive environment.

## 🧠 Key Features

- **12x9 Dynamic Grid**: Interactive crossword board with real-time updates
- **Minimax AI**: Intelligent opponent using game tree search
- **Dual Game Modes**: Human vs Human and AI simulation
- **Strategic Scoring**: Point-based system rewarding valid placements
- **Turn-based Logic**: Alternating player moves with state management

## 🎯 Learning Objectives

- **Game Theory**: Understanding zero-sum games and optimal strategies
- **Search Algorithms**: Minimax implementation with depth-limited search
- **State Space**: Game state representation and evaluation
- **AI Decision Making**: Automated move selection and strategy

## 🏗️ Technical Implementation

### Core Components
- **Grid Management**: Dynamic board state tracking
- **Move Validation**: Word placement and crossing rules
- **Score Calculation**: Real-time point tracking system
- **AI Engine**: Minimax algorithm with game tree exploration

### Algorithm Details
```python
def minimax(board, depth, maximizing_player):
    # Evaluate terminal states
    # Generate possible moves
    # Recursively explore game tree
    # Return optimal move value
```

## 🚀 How to Run

```bash
# Navigate to project directory
cd semester-1/game-ai-minimax/

# Run interactive mode (Human vs Human)
python crossword_game.py --mode interactive

# Run AI simulation mode
python crossword_game.py --mode ai
```

## 📊 Game Rules & Scoring

### Placement Rules
- First word "RABBIT" placed at fixed position (5,2)
- New words must intersect with existing words
- Valid crossings required for placement

### Scoring System
- **Valid Placement**: +1 point per letter
- **Invalid Move**: -1 point penalty
- **Game End**: When no valid moves remain

## 🎯 AI Strategy

The Minimax algorithm evaluates:
1. **Board Control**: Strategic position occupation
2. **Scoring Potential**: Maximum point opportunities
3. **Opponent Blocking**: Limiting opponent's options
4. **Endgame Planning**: Optimal final moves

## 📈 Results & Analysis

### Interactive Mode
- Real-time player decision making
- Strategic thinking development
- Human vs human competition

### AI Simulation
- 10 automated game iterations
- Best scoring outcomes analysis
- Algorithm performance evaluation

## 🔧 Technical Skills Demonstrated

- **Algorithm Design**: Minimax implementation
- **Game Theory**: Strategic decision making
- **Python Programming**: Object-oriented design
- **Problem Solving**: Complex state management
- **Performance Optimization**: Efficient search pruning

## 🎓 Academic Value

This project showcases understanding of:
- **Artificial Intelligence**: Search algorithms and decision trees
- **Computer Science**: Algorithm complexity and optimization
- **Game Theory**: Strategic thinking and optimal play
- **Software Engineering**: Clean code and modular design

---

*Part of M.Tech AI/ML Academic Portfolio - BITS Pilani*