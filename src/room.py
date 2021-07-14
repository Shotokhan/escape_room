import random
from config import Config
from os import linesep
import json


class Room:
    def __init__(self, config):
        self.rows = config['rows']
        self.columns = config['columns']
        self.bots = config['bots']
        self.map_chars = config['map_chars']
        self.labyrinth = self.create_labyrinth()

    def create_labyrinth(self):
        lab = [[self.map_chars['free_boot'] for _ in range(self.columns)] for _ in range(self.rows)]
        free = set([(i, j) for i in range(self.rows) for j in range(self.columns)])
        for i in range(self.rows):
            lab[i][0] = self.map_chars['wall']
            free.remove((i, 0))
            lab[i][self.columns - 1] = self.map_chars['wall']
            free.remove((i, self.columns - 1))
        for i in range(self.columns):
            lab[0][i] = self.map_chars['wall']
            try:
                free.remove((0, i))
            except KeyError:
                pass
            lab[self.rows - 1][i] = self.map_chars['wall']
            try:
                free.remove((self.rows - 1, i))
            except KeyError:
                pass
        vertical = random.choice([0, 1])
        if vertical:
            rand = random.choice([i for i in range(1, self.rows - 1)])
            side = random.choice([0, self.columns - 1])
            lab[rand][side] = self.map_chars['escape']
        else:
            rand = random.choice([i for i in range(1, self.columns - 1)])
            side = random.choice([0, self.rows - 1])
            lab[side][rand] = self.map_chars['escape']
        player = random.choice(list(free))
        lab[player[0]][player[1]] = self.map_chars['player']
        free.remove(player)
        num_walls = min(self.rows, self.columns)
        for _ in range(num_walls):
            wall = random.choice(list(free))
            lab[wall[0]][wall[1]] = self.map_chars['wall']
            free.remove(wall)
        for _ in range(self.bots):
            bot = random.choice(list(free))
            lab[bot[0]][bot[1]] = self.map_chars['monster']
            free.remove(bot)
        return lab

    def __str__(self):
        return linesep.join(["".join([i for i in row]) for row in self.labyrinth])

    def jsonify(self):
        return json.dumps({"rows": self.rows, "columns": self.columns, "bots": self.bots,
                           "map_chars": self.map_chars, "labyrinth": self.__str__()})


if __name__ == "__main__":
    conf = Config()
    room = Room(conf)
    print(room)
    j = room.jsonify()
    print(j)
