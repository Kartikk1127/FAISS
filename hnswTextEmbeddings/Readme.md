# Hierarchical Navigable Small World

These are one of the highest performing indexes.  

_Using the example of Facebook — in 2016, we could connect every user (a vertex) to their Facebook friends (their nearest neighbors). And despite the 1.59B active users, the average number of steps (or hops) needed to traverse the graph from one user to another was just 3.57 [2]._

At a high level, HNSW graphs are built by taking NSW graphs and breaking them apart into multiple layers. With each incremental layer eliminating intermediate connection between verticles.

For bigger datasets with higher-dimensionality, HNSW graphs are some of the best performing indexes we can use. And by layering other quantization steps, we can also improve search-times even further.  


## Real life analogy
Think of HNSW as building a network of friends to help you find things quickly:
M = 8 (How many friends each person has)

Every person in the network knows exactly 8 other people
More friends = easier to find what you're looking for, but costs more to maintain all those friendships
M=8 means each data point connects to 8 others

ef_construction = 32 (How hard you try when making new friends)

When a new person joins the network, they look at 32 potential friends before deciding which 8 to actually befriend
Higher number = you're pickier about friends = better network quality, but takes longer to set up

ef_search = 32 (How many people you ask when looking for something)

When searching, you ask 32 people "do you know where to find X?"
More people asked = better chance of finding the right answer, but takes more time

Simple analogy:
Imagine you're looking for a book in a city:

Flat index: Ask everyone in the city (slow but guaranteed to find it)
HNSW: Ask your 8 friends, who ask their 8 friends, etc. (much faster, usually finds it)

The parameters control how well-connected this friend network is. Your values (M=8, ef=32) create a moderately connected network that should be fast and accurate without using too much memory.