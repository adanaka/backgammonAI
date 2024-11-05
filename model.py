import os
import tensorflow as tf

from backgammon.agents.human_agent import HumanAgent
from backgammon.agents.random_agent import RandomAgent
from backgammon.game import Game
from backgammon.agents.td_gammon_agent import TDAgent
import random
from backgammon.agents.eater_agent import EaterAgent
from backgammon.agents.close_agent import CloseAgent
from backgammon.agents.expecti_mm_agent import ExpectMinMaxAgent
from backgammon.heuristics import HeuristicEvaluator


class Model(tf.keras.Model):
    def __init__(self, model_path, summary_path, checkpoint_path, restore=False):
        super(Model, self).__init__()
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path

        # Define the network
        layer_size_hidden = 50
        layer_size_output = 1

        self.dense1 = tf.keras.layers.Dense(layer_size_hidden, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(layer_size_output, activation='sigmoid')

        # Learning rate decay
        self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.1,
            decay_steps=40000,
            decay_rate=0.96,
            staircase=True
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule)

        # Metrics
        self.loss_metric = tf.keras.metrics.Mean()
        self.delta_metric = tf.keras.metrics.Mean()
        self.accuracy_metric = tf.keras.metrics.Mean()

        # Define Checkpoint and Manager
        self.checkpoint = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=5)

        # Restore from checkpoint if needed
        if restore:
            self.restore()

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    @tf.function
    def train_step(self, x, V_next):
        with tf.GradientTape() as tape:
            V = self(x, training=True)
            delta = tf.reduce_sum(V_next - V)
            loss = tf.reduce_mean(tf.square(V_next - V))
            accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.round(V_next), tf.round(V)), dtype=tf.float32))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.loss_metric.update_state(loss)
        self.delta_metric.update_state(delta)
        self.accuracy_metric.update_state(accuracy)

    def train(self):
        summary_writer = tf.summary.create_file_writer(self.summary_path)

        # The agent plays against itself
        players = [TDAgent(Game.TOKENS[0], self), TDAgent(Game.TOKENS[1], self)]

        episodes = 100

        for episode in range(episodes):

            game = Game.new()
            player_num = random.randint(0, 1)

            x = game.extract_features(players[player_num].player)

            game_step = 0
            while not game.is_over():
                game.next_step(players[player_num], player_num)
                player_num = (player_num + 1) % 2

                x_next = game.extract_features(players[player_num].player)

                V_next = self(x_next, training=False)
                self.train_step(x, V_next)

                x = x_next
                game_step += 1

            winner = game.winner()

            with summary_writer.as_default():
                tf.summary.scalar('loss', self.loss_metric.result(), step=episode)
                tf.summary.scalar('delta', self.delta_metric.result(), step=episode)
                tf.summary.scalar('accuracy', self.accuracy_metric.result(), step=episode)

            print(f"Game {episode}/{episodes} (Winner: {players[winner].player}) in {game_step} turns")

            # Save the model weights manually
            self.checkpoint_manager.save()

        summary_writer.close()

    def play(self, num_games=10):
        agent1_wins = 0
        agent2_wins = 0

        game = Game.new()

        agent_1 = ExpectMinMaxAgent(1, Game.TOKENS[0], HeuristicEvaluator(game, 0))
        agent_2 = TDAgent(Game.TOKENS[1], self)

        for i in range(num_games):

            winner = game.play([agent_1, agent_2], draw=False)

            if winner == 0:
                agent1_wins += 1
            elif winner == 1:
                agent2_wins += 1

            print(f"Game {i + 1}: Winner is {winner}")

            game = Game.new()

        print(f"\nSummary after {num_games} games:")
        print(f"{agent_1.name} Agent wins: {agent1_wins}")
        print(f"{agent_2.name} Agent wins: {agent2_wins}")

    def restore(self):
        # Restore the latest checkpoint
        latest_checkpoint_path = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.checkpoint.restore(latest_checkpoint_path).expect_partial()
        else:
            print('No checkpoint found.')
