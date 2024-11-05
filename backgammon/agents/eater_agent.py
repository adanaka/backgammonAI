import random

# The agent will consider all valid moves.
# It will first try to find a move where it can land on a point occupied by a single opponent piece,
# allowing it to send that piece to the bar.
# If no such move exists, it will make a random or default move from the available options.


class EaterAgent:
    def __init__(self, player_token):
        """
        Initialize the EaterAgent with a token ('x' or 'o').
        """
        self.player = player_token
        self.name = 'Eater'

    def get_action(self, moves, game):
        """
        Chooses a move that tries to eat the opponent's piece, if possible.
        If no eating move is available, it returns a random move.
        """
        # If no moves are available, return None
        if not moves:
            return None

        # Token of the opponent
        opponent_token = game.opponent(self.player)

        # First try to find an eating move
        for move in moves:
            for start, end in move:
                # Check if the end point has exactly one opponent piece
                if end != 'off' and len(game.grid[end]) == 1 and game.grid[end][0] == opponent_token:
                    return move  # This is an eating move

        # If no eating move is found, return a random move
        return random.choice(list(moves))
