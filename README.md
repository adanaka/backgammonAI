
Training the TD Model:

To train the TD model for n games, follow these steps:

1) In model.py, set the episodes variable to n in the train() function.

2) In main.py, set the checkpoint_path variable (at line 16) to the desired file path where you want to
 save the model weights.

3) After that, run the following command: python main.py



Playing:

In our project, we implemented six different agents. To run n games between two of these agents, follow these steps:

1) go to the play(num_games) function in model.py.

2) Define agent_1 and agent_2 variables as instances of the agent objects (e.g., RandomAgent, TDAgent, etc.).

3) Next, go to main.py and add n as an argument to the model.play(n) function call.

4) Finally, run the following command to start the games: python main.py --play --restore.


Notes:
1) If using Human agent, If a checker is eaten and you want to return it to the board, input: on,x , where x is
 the point on the board that you can reach. If a checker wants to board off, input: x,off , where x is the point
  from which you are bearing off.
2) if one of the players is a TD agent, ensure that the checkpoint_path variable (at line 16 in main.py) is set to
 the file path where the model weights are saved.
