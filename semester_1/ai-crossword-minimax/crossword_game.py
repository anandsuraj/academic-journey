import copy
import random

def print_grid(grid):
    # Print the grid with row and column numbers for better visualization
    print("   " + " ".join(str(i) for i in range(len(grid[0]))))  # Column numbers
    for i, row in enumerate(grid):
        print(f"{i:2} " + ' '.join(row))  # Row numbers
    print()

class CrosswordGame:
    def __init__(self, words):
        # Initialize the crossword grid and game variables
        self.rows, self.cols = 12, 9
        self.grid = [['#' for _ in range(self.cols)] for _ in range(self.rows)]
        self.words = words
        self.remaining_words = words.copy()
        self.player_scores = [0, 0]
        self.current_player = 1
        self.placed_words = []

        # Place first word (RABBIT) at 1,5,D
        if 'RABBIT' in self.remaining_words:
            start_row = 1
            start_col = 5
            if self.can_place_word('RABBIT', start_row, start_col, 'D'):
                self.place_first_word('RABBIT', start_row, start_col, 'D')
                self.player_scores[0] += len('RABBIT')
                self.remaining_words.remove('RABBIT')
                self.placed_words.append(('RABBIT', start_row, start_col, 'D'))
                # print(f"Player 1 placed RABBIT at ({start_row}, {start_col}) direction D")
                # print(f"Score: Player 1: {self.player_scores[0]}, Player 2: {self.player_scores[1]}")
                self.current_player = 2
            else:
                print("Could not place RABBIT at (1,5) direction D. Invalid initial placement.")

    def place_first_word(self, word, row, col, direction):
        # Special method to place the first word (RABBIT)
        if direction == 'D':
            for i in range(len(word)):
                self.grid[row + i][col] = word[i]
        else:
            for i in range(len(word)):
                self.grid[row][col + i] = word[i]

    def has_intersection(self, word, row, col, direction):
        # Check if the word intersects with any existing word
        has_connection = False

        if direction == 'A':
            for i in range(len(word)):
                # Check if this position already has a letter (intersection)
                if self.grid[row][col + i] != '#' and self.grid[row][col + i] == word[i]:
                    has_connection = True

                # Check adjacent cells (above and below)
                if row > 0 and self.grid[row - 1][col + i] != '#':
                    has_connection = True
                if row < self.rows - 1 and self.grid[row + 1][col + i] != '#':
                    has_connection = True

        elif direction == 'D':
            for i in range(len(word)):
                # Check if this position already has a letter (intersection)
                if self.grid[row + i][col] != '#' and self.grid[row + i][col] == word[i]:
                    has_connection = True

                # Check adjacent cells (left and right)
                if col > 0 and self.grid[row + i][col - 1] != '#':
                    has_connection = True
                if col < self.cols - 1 and self.grid[row + i][col + 1] != '#':
                    has_connection = True

        return has_connection

    def can_place_word(self, word, row, col, direction):
        # Check if the word can be placed at the given position

        if direction == 'A':
            if col + len(word) > self.cols:
                return False
            if col < 0:
                return False
            for i in range(len(word)):
                if col + i >= self.cols:
                    return False
                if self.grid[row][col + i] != '#' and self.grid[row][col + i] != word[i]:
                    return False
        elif direction == 'D':
            if row + len(word) > self.rows:
                return False
            if row < 0:
                return False

            for i in range(len(word)):
                 if row + i >= self.rows:
                    return False
                 if self.grid[row + i][col] != '#' and self.grid[row + i][col] != word[i]:
                    return False
        else:
            return False  # Invalid direction

        return True

    def place_word(self, word, row, col, direction):
        # Place the word in the grid if it's valid
        if not self.can_place_word(word, row, col, direction):
            return False

        if direction == 'A':
            for i in range(len(word)):
                self.grid[row][col + i] = word[i]
        elif direction == 'D':
            for i in range(len(word)):
                self.grid[row + i][col] = word[i]

        self.placed_words.append((word, row, col, direction))
        return True

    def play(self, auto_mode=False):
        # Main game loop

        if not auto_mode:
            print("Game started with RABBIT placed at (1,5) direction D")
            print_grid(self.grid)

        while self.remaining_words:
            if not auto_mode:
                # Player input mode (Manual input for two players)
                print(f"Player {self.current_player}'s turn")
                print("Available words:", self.remaining_words)

                word = input("Choose a word: ").strip().upper()
                if word not in self.remaining_words:
                    print("Invalid word. Please choose from the available words.")
                    continue

                try:
                    row = int(input(f"Enter row (0-{self.rows - 1}): ").strip())
                    col = int(input(f"Enter column (0-{self.cols - 1}): ").strip())
                except ValueError:
                    print("Invalid row or column. Please enter a number.")
                    continue

                direction = input("Enter direction (A for Across, D for Down): ").strip().upper()

                if self.place_word(word, row, col, direction):
                    self.player_scores[self.current_player - 1] += len(word)
                    self.remaining_words.remove(word)
                    print(f"Player {self.current_player} placed {word} at ({row}, {col}) direction {direction}")
                    print(f"Score: Player 1: {self.player_scores[0]}, Player 2: {self.player_scores[1]}")
                    print_grid(self.grid)
                else:
                    self.player_scores[self.current_player - 1] -= 1  # always loses 1 point
                    print("Invalid placement! Player loses 1 point.")
                    print(f"Score: Player 1: {self.player_scores[0]}, Player 2: {self.player_scores[1]}")

                self.current_player = 2 if self.current_player == 1 else 1

            else:
                # AI mode
                best_move = self.minimax_alpha_beta(self.current_player, depth=3, alpha=float('-inf'), beta=float('inf'))
                if best_move:
                    word, row, col, direction = best_move
                    if self.place_word(word, row, col, direction):
                        self.player_scores[self.current_player - 1] += len(word)
                        self.remaining_words.remove(word)
                        self.current_player = 2 if self.current_player == 1 else 1  # Switch player
                    else:
                        print("Minimax returned invalid move!")
                        self.player_scores[self.current_player - 1] -= 1
                else:
                    print("No valid moves found by Minimax!")
                    break # No moves are available


        if not auto_mode:
            print("Game Over!")
            print(f"Final Scores: Player 1: {self.player_scores[0]}, Player 2: {self.player_scores[1]}")
            if self.player_scores[0] > self.player_scores[1]:
                print("Player 1 wins!")
            elif self.player_scores[1] > self.player_scores[0]:
                print("Player 2 wins!")
            else:
                print("It's a tie!")

        return self.player_scores, self.grid

    def minimax_alpha_beta(self, player, depth, alpha, beta):
        # Implements the Minimax algorithm with alpha-beta pruning

        if depth == 0 or not self.remaining_words:
            return self.evaluate_board_for_minimax(player), None, None, None # Return the evaluation, move is None

        if player == self.current_player:  # Maximizing player
            best_score = float('-inf')
            best_move = None

            valid_moves = self.generate_valid_moves()

            for move in valid_moves:
                word, row, col, direction = move
                # Simulate the move
                temp_game = copy.deepcopy(self)
                temp_game.place_word(word, row, col, direction)
                temp_game.remaining_words.remove(word)
                temp_game.current_player = 2 if temp_game.current_player == 1 else 1 # Switch player

                # Get score from the minimizing player
                score, _, _, _ = temp_game.minimax_alpha_beta(3 - player, depth - 1, alpha, beta) # 3 - player to switch between 1 and 2

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return best_score, best_move[0] if best_move else None, best_move[1] if best_move else None, best_move[2] if best_move else None # Return the best move

        else: # Minimizing player
            best_score = float('inf')
            best_move = None

            valid_moves = self.generate_valid_moves()

            for move in valid_moves:
                word, row, col, direction = move
                # Simulate the move
                temp_game = copy.deepcopy(self)
                temp_game.place_word(word, row, col, direction)
                temp_game.remaining_words.remove(word)
                temp_game.current_player = 2 if temp_game.current_player == 1 else 1 # Switch player

                # Get score from the maximizing player
                score, _, _, _ = temp_game.minimax_alpha_beta(self.current_player, depth - 1, alpha, beta)

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return best_score, best_move[0] if best_move else None, best_move[1] if best_move else None, best_move[2] if best_move else None # Return the best move

    def generate_valid_moves(self):
        valid_moves = []
        for word in self.remaining_words:
            for row in range(self.rows):
                for col in range(self.cols):
                    for direction in ['A', 'D']:
                        if self.can_place_word(word, row, col, direction):
                            valid_moves.append((word, row, col, direction))
        return valid_moves

    def evaluate_board_for_minimax(self, player):
        # Evaluation function for Minimax
        # Assign a numerical score to a given board state.

        if player == 1:
            return self.player_scores[0] - self.player_scores[1]
        else:
            return self.player_scores[1] - self.player_scores[0]

    def count_intersections(self, word, row, col, direction):
        # Count how many letters would intersect with existing words
        intersections = 0

        if direction == 'A':
            for i in range(len(word)):
                if self.grid[row][col + i] != '#':
                    intersections += 1
        else:  # direction == 'D'
            for i in range(len(word)):
                if self.grid[row + i][col] != '#':
                    intersections += 1

        return intersections

def find_best_solutions(words):
    # Run 10 iterations to find the two best solutions
    solutions = []
    for i in range(10):
        print(f"Running iteration {i+1}/10...")
        game = CrosswordGame(words.copy())  # Pass in the correct size
        scores, final_grid = game.play(auto_mode=True)
        solutions.append((scores, final_grid))
        # print("Current Grid:") # Added grid output in auto mode
        # print_grid(final_grid)

    # Sort solutions based on total score
    solutions.sort(key=lambda x: sum(x[0]), reverse=True)
    best_solutions = solutions[:2]

    print("\nTwo best solutions:")
    for idx, (scores, grid) in enumerate(best_solutions, 1):
        print(f"Solution {idx}:")
        print(f"Scores: Player 1: {scores[0]}, Player 2: {scores[1]}")
        print("Final Grid:")
        print_grid(grid)
        print("Final variable values:")
        print(f"Player scores: {scores}")
        print(f"Total score: {sum(scores)}")
        print()

def main():
    # Main function to start the game with user choice
    words = ['CAT', 'DOG', 'RABBIT', 'ELEPHANT', 'MONKEY', 'HORSE', 'CAMEL', 'DONKEY']
    print("Two-player Solution-Based Crossword Puzzle")
    print("------------------------------------------")
    print("\nGame Modes:")
    print("1. Two Player Interactive Mode")
    print("2. AI Best Solutions Mode")
    print("-" * 30)
    mode = input("\nSelect your game mode (1/2): ").strip()

    if mode == '1':
        # Player-vs-Player mode (Interactive gameplay)
        game = CrosswordGame(words) 
        game.play()
    elif mode == '2':
        # AI-driven mode (Finding best solutions through iterations)
        find_best_solutions(words)
    else:
        print("Invalid choice. Please restart and enter 1 or 2.")

if __name__ == "__main__":
    main()
