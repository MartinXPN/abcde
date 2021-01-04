from data import RandomGraphs
from util import fix_random_seed

fix_random_seed(42)
data = RandomGraphs(min_nodes=10, max_nodes=11, nb_graphs=1)

g = data[0]
# X [0.5   0.2    0.6    0.4    0.8    0.4    0.5    0.5    0.5    0.4]
print('X', g.x.squeeze(-1))
# Y [1.33445375 0.         1.18216541 1.5785789  0.62944579 1.66235589
#  1.1196099  1.43682666 1.25527251 1.69100107]
print('Y', g.y.squeeze(-1))
