from nim import NimAI, Nim

ai = NimAI()

game = Nim()    
# ai.q[(0, 0, 0, 2), (3, 2)]= -1

game.piles = (0, 0, 0, 2)

print(f"----ai q= {ai.q}")
print(f"---game piles= {game.piles}")
print(f"---actions={game.available_actions((0,0,0,2))}")
print(f"---bestfuture={ai.best_future_reward((0, 0, 0, 2))}")
action = (3,1)
ai.update((0, 0, 0, 2), action, (0, 0, 0, 1), 1)
# ai.q[(0, 0, 0, 2), (3, 1)]= 1
print(f"---ai q= {ai.q}")
print(f"---aiq(n)={ai.get_q_value((0, 0, 0, 2), (3, 1))})")

