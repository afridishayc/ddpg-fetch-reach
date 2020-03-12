
from critic import Critic
from actor import Actor

actor_ = Actor()
actor_network = actor_.create_network(2, 1, 2, perceptrons_count=128)
target_actor_network = actor_.create_network(2, 1, 2, perceptrons_count=64)

print(actor_network.summary())
print(target_actor_network.summary())