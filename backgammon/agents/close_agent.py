class CloseAgent:
    def __init__(self, player_token):
        self.player = player_token
        self.name = 'Close'

    def get_action(self, moves, game):
        best_move = None
        best_score = -float('inf')

        for move in moves:
            score = self.evaluate_move(move, game)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def evaluate_move(self, move, game):
        """
        choose a move based on the agent's strategy to build blocks and prevent opponent from eating pieces.
        """

        temp_game = game.clone()
        temp_game.take_action(move, self.player)

        score = 0

        # Check the number of building blocks
        for col in temp_game.grid:
            if len(col) > 1 and col[0] == self.player:
                score += len(col)

        return score
