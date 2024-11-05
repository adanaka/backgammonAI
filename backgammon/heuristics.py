

class HeuristicEvaluator:
    """
    Terminology:
    * Blot - A single checker sitting alone on a point where it is vulnerable to being hit.
    * Hit - To move to a point occupied by an opposing blot and put the blot on the bar.
    * Block - A point occupied by two or more checkers of the same color.
    * Blockade - A series of blocks arranged to prevent escape of the opponent's runners.
    * Prime - Several consecutive blocks
    * Anchor - A block in the opponent's home board.
    """

    def __init__(self, game, player):
        self.player = game.TOKENS[player]
        self.opponent = game.TOKENS[1 - player]
        self.quadrants = None
        self.game = game

    def evaluate(self, game) -> float:

        self.game = game
        self._fill_quadrants()

        return sum(feature * weight for feature, weight in [
            (self._vulnerability_score(), 0.4),
            (self._hitting_score(), 0.4),
            (self._blocking_score(), 0.01),
            (self._bear_in_score(), 0.4),
            (self._bear_off_score(), 0.4),
            (self._terminal_state_score(), 1)]) - self.opponent_score()

    def opponent_score(self) -> float:
        temp = self.player
        self.player = self.opponent
        self.opponent = temp
        self._fill_quadrants()
        score = sum(feature * weight for feature, weight in [
            (self._hitting_score(), 0.4),
            (self._blocking_score(), 0.4),
            (self._bear_in_score(), 0.2),
            (self._bear_off_score(), 0.2),
            (self._terminal_state_score(), 1),
        ])
        temp = self.player
        self.player = self.opponent
        self.opponent = temp
        return score

    def _vulnerability_score(self) -> float:
        """ Minimize the amount of blots, based on quadrants."""
        if self.player == 'o':
            score = sum([self._count_blots(self.quadrants[i]) * i for i in range(0, 4)])
        else:
            score = sum([self._count_blots(self.quadrants[i]) * (3-i) for i in range(0, 4)])
        return 1 - self._normalize(score, 0, 15)

    def _terminal_state_score(self) -> float:
        """ Evaluates a terminal state """
        if len(self.game.off_pieces[self.player]) == 15:
            return float('inf')
        if len(self.game.off_pieces[self.opponent]) == 15:
            return float('-inf')
        else:
            return 0

    def _hitting_score(self) -> float:
        """ Maximize the amount of the eaten opponent's checkers """
        score = len(self.game.bar_pieces[self.opponent]) - len(self.game.bar_pieces[self.player])/2
        return self._normalize(score, 0, 15)

    def _blocking_score(self) -> float:
        """ Maximize the amount of blocks, based on quadrants. """
        if self.player == 'o':
            score = sum([self._count_blocks(self.quadrants[i]) * i for i in range(0, 4)])
        else:
            score = sum([self._count_blocks(self.quadrants[i]) * (3-i) for i in range(0, 4)])  # TODO: consider anchors
        return self._normalize(score, 0, 27)

    def _bear_in_score(self):
        """ Maximize the amount of checkers inside home board"""
        score_x = sum([sum([1 for point in col if point == self.opponent]) for col in self.quadrants[0]])
        score_o = sum([sum([1 for point in col if point == self.opponent]) for col in self.quadrants[3]])
        if self.player == 'o':
            score = sum([sum([1 for point in col if point == self.player]) for col in self.quadrants[3]]) - score_x/2
        else:
            score = sum([sum([1 for point in col if point == self.player]) for col in self.quadrants[0]]) - score_o/2
        return self._normalize(score, 0, 15)

    def _bear_off_score(self):
        """ Maximize the amount of checkers inside home board"""
        score = len(self.game.off_pieces[self.player]) - len(self.game.off_pieces[self.opponent])
        return self._normalize(score, 0, 15)

    def _fill_quadrants(self):
        grid = self.game.grid
        self.quadrants = [
            grid[0:6],
            grid[6:12],
            grid[12:18],
            grid[18:24],
        ]

    def _count_blocks(self, quarter):
        return sum([sum([1 for point in col if point == self.player]) for col in quarter if sum([1 for point in col if point == self.player]) > 1])

    def _count_blots(self, quarter):
        return sum([sum([1 for point in col if point == self.player]) for col in quarter])

    @staticmethod
    def _normalize(x: float, x_min: float, x_max: float) -> float:
        """
        Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1.
        It is also known as Min-Max scaling.
        :param x: the actual feature value.
        :param x_max: the maximum possible value of the feature.
        :param x_min: the minimum possible value of the feature.
        :return: A number between 0 and 1.
        """
        return (x - x_min) / (x_max - x_min)