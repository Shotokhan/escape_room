from room import linesep
import random


def get_matrix(room_repr):
    return room_repr.split(linesep)


def get_legal_positions(room_mat, rows, columns, wall):
    legal = set()
    for i in range(rows):
        for j in range(columns):
            if room_mat[i][j] != wall:
                legal.add((i, j))
    return legal


def get_special_positions(room_mat, rows, columns, escape, monster, player):
    objects = {}
    counter = 1
    for i in range(rows):
        for j in range(columns):
            if room_mat[i][j] == monster:
                objects["monster_{}".format(counter)] = (i, j)
                counter += 1
            elif room_mat[i][j] == escape:
                objects["escape"] = (i, j)
            elif room_mat[i][j] == player:
                objects["player"] = (i, j)
    return objects


def change_character(mat, row_ind, col_ind, new_char):
    row = [i for i in mat[row_ind]]
    row[col_ind] = new_char
    return "".join(row)


def move(objects, entity, delta, legal):
    old_pos = objects[entity]
    new_pos = (old_pos[0] + delta[0], old_pos[1] + delta[1])
    if new_pos in legal:
        objects[entity] = new_pos
        return True
    else:
        return False


def add_buttons(s):
    s += "<br> <br>"
    s += '<form action="" method="post"><input type="submit" name="action" value="move_up" ' \
         '/></form><br>'
    s += '<form action="" method="post"><input type="submit" name="action" value="move_down" ' \
         '/></form><br>'
    s += '<form action="" method="post"><input type="submit" name="action" value="move_right" ' \
         '/></form><br>'
    s += '<form action="" method="post"><input type="submit" name="action" value="move_left" ' \
         '/></form><br>'
    s += '<form action="" method="post"><input type="submit" name="action" value="no_action" ' \
         '/></form><br>'
    return s


def transition_function(session, rows, columns, map_chars, player_action, bots):
    legal = get_legal_positions(session['room'], rows, columns, map_chars['wall'])
    transition = {
        "move_up": lambda entity: move(session, entity, (-1, 0), legal),
        "move_down": lambda entity: move(session, entity, (1, 0), legal),
        "move_right": lambda entity: move(session, entity, (0, 1), legal),
        "move_left": lambda entity: move(session, entity, (0, -1), legal),
        "no_action": lambda entity: True,
        "lol_nope": lambda entity: True
    }
    actions = {}
    old_pos = {"monster_{}".format(i+1): session["monster_{}".format(i+1)] for i in range(bots)}
    old_pos['player'] = session['player']
    if player_action not in transition.keys():
        player_action = "lol_nope"
    result = transition[player_action]('player')
    if not result:
        actions['player_action'] = "lol_nope"
    else:
        actions['player_action'] = player_action
        session['room'][old_pos['player'][0]] = change_character(session['room'], old_pos['player'][0],
                                                                 old_pos['player'][1], map_chars['free_boot'])
        new_pos = session['player']
        session['room'][new_pos[0]] = change_character(session['room'], new_pos[0], new_pos[1], map_chars['player'])
    for i in range(bots):
        bot = "monster_{}".format(i + 1)
        action = random.choice(list(transition.keys() - "lol_nope"))
        result = transition[action](bot)
        if not result or old_pos[bot] == session['player']:
            actions[bot] = 'no_action'
        else:
            new_pos = session[bot]
            if new_pos == session['escape']:
                actions[bot] = 'no_action'
                session[bot] = old_pos[bot]
            else:
                actions[bot] = action
                session['room'][old_pos[bot][0]] = change_character(session['room'], old_pos[bot][0],
                                                                    old_pos[bot][1], map_chars['free_boot'])
                session['room'][new_pos[0]] = change_character(session['room'], new_pos[0], new_pos[1],
                                                               map_chars['monster'])
    return actions
