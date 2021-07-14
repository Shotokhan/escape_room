import json
from datetime import timedelta
from os import linesep

from flask import Flask, request, session, redirect

from config import Config
from room import Room
from util_room import get_matrix, get_special_positions, add_buttons, transition_function

app = Flask(__name__)
app.config['SECRET_KEY'] = 'can_you_escape_the_room'
config = Config()
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=config['flask']['permanent_session_seconds'])
flag = "naCTF{C4n_you_3sc4p3_th3_2337_r00m}"
with open('problem_statement', 'r') as f:
    statement = f.read()


@app.route('/', methods=['GET'])
def index():
    return statement


@app.route('/problem', methods=['GET', 'POST'])
def problem():
    if request.method == 'GET':
        if 'player' in session:
            if request.args.to_dict().get("pretty") == "true":
                s = json.dumps({key: session[key] for key in list(session.keys() -
                                                                  ("legal", "_permanent"))}) + "<br> <br>"
                for row in session['room']:
                    s += row + "<br>"
                if config['test']:
                    s = add_buttons(s)
                return s
            else:
                return json.dumps({key: session[key] for key in list(session.keys() - ("legal", "_permanent"))})
        else:
            room = Room(config)
            session.permanent = True
            room_mat = get_matrix(str(room))
            session['room'] = room_mat
            objects = get_special_positions(room_mat, room.rows, room.columns, room.map_chars['escape'],
                                            room.map_chars['monster'], room.map_chars['player'])
            session['player'] = objects['player']
            for i in range(room.bots):
                session['monster_{}'.format(i+1)] = objects['monster_{}'.format(i+1)]
            session['escape'] = objects['escape']
            if request.args.to_dict().get("pretty") == "true":
                s = room.jsonify() + "<br> <br>"
                s += str(room).replace(linesep, "<br>")
                if config['test']:
                    s = add_buttons(s)
                return s
            else:
                return room.jsonify()
    elif request.method == 'POST':
        if 'player' in session:
            params = request.form.to_dict()
            if "action" not in list(params.keys()):
                action = "lol_nope"
            else:
                action = params['action']
            actions = transition_function(session, config['rows'], config['columns'],
                                          config['map_chars'], action, config['bots'])
            for i in range(config['bots']):
                if session['monster_{}'.format(i+1)] == session['player']:
                    [session.pop(key) for key in list(session.keys())]
                    return "Wasted!"
            if session['player'] == session['escape']:
                [session.pop(key) for key in list(session.keys())]
                return flag
            else:
                if config['test']:
                    return redirect("/problem?pretty=true")
                else:
                    return json.dumps(actions)
        else:
            if config['test']:
                return redirect("/problem?pretty=true")
            else:
                return redirect("/problem")


if __name__ == "__main__":
    app.debug = config['flask']['debug']
    app.run(host=config['flask']['host'], port=config['flask']['port'])
