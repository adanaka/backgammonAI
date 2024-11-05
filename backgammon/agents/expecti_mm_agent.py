import math
PRUNING_FACTOR = 10
class ExpectMinMaxAgent:
    def __init__(self, depth=1, player='x', heuristic=None):
        self.heuristic = heuristic
        self.depth = depth  # Depth of search tree
        self.factor = 1/36  # Factor to normalize the expected value of dice rolls
        self.player = player
        self.opponent = 'o' if player == 'x' else 'x'
        self.name = 'ExpectiMinMax'

    def expectiminimax(self, game, depth, maximizingPlayer, chance_node, player, moves):
        """
        Core expectiminimax function:
        - game: the current game state
        - depth: depth of the search tree
        - maximizingPlayer: True if we are maximizing, False if minimizing
        """
        if depth <= 0 or game.is_over():
            if chance_node:
                return self.evaluate(game, self.player)
            return self.evaluate(game, player)

        # Handle chance nodes (dice roll or random events)
        if chance_node:
            return self.handle_chance_node(game, depth, player, maximizingPlayer)

        if maximizingPlayer:
            maxEval = -math.inf
            for move in moves:  # Assuming 'x' is the agent's token
                ate_list = game.take_action(move, player)
                eval = self.expectiminimax(game, depth - 1, not maximizingPlayer, True, self.opponent, None)
                maxEval = max(maxEval, eval)
                game.undo_action(move, player, ate_list)
            return maxEval

        else:
            minEval = math.inf
            for move in moves:
                ate_lst = game.take_action(move, player)
                eval = self.expectiminimax(game, depth - 1, not maximizingPlayer, True, self.player, None)
                minEval = min(minEval, eval)
                game.undo_action(move, player, ate_lst)
            return minEval

    def handle_chance_node(self, game, depth, player, maximizingPlayer):
        """
        Handle chance nodes (random events like dice rolls).
        We return the expected value of all possible outcomes.
        """
        expected_value = 0
        for dice, factor in [(t, 2)for t in game.get_possible_rolls_excluding_doubles()] + [(t, 1) for t in game.get_possible_doubles()]:
            moves = game.get_actions(dice, player)
            if len(moves) > PRUNING_FACTOR:
                depth = 0
            dice_value = self.expectiminimax(game, depth, maximizingPlayer, False, player, moves) * factor
            expected_value += dice_value * self.factor
        return expected_value

    def evaluate(self, game, player=None):
        """
        Evaluate the game state.
        You will need to implement a custom evaluation function based on game rules.
        """
        if player != self.player:
            return self.heuristic.evaluate(game)
        else:
            return self.heuristic.evaluate(game)

    def get_action(self, moves, game=None):
        """
        Returns the best action for the agent by evaluating all possible moves using expectiminimax.
        Parameters:
        - moves: a list of all legal moves
        - game: the current game state
        Returns:
        - The best move based on the expectiminimax algorithm
        """
        best_move = None
        best_value = -math.inf
        depth = self.depth

        # Loop over all available moves
        for move in moves:
            # Simulate the move
            ateList = game.take_action(move, self.player)
            if len(moves) > PRUNING_FACTOR:
                depth = 0
            else:
                depth = self.depth
            # Run expectiminimax to evaluate this move
            value = self.expectiminimax(game, depth - 1, False, True,
                                        self.opponent, None)

            # Undo the simulated move
            game.undo_action(move, self.player, ateList)

            # Maximizing the score for the agent
            if value > best_value:
                best_value = value
                best_move = move
        return best_move