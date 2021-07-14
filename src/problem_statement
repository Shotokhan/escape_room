<title>Escape room</title>
<strong>escape_room</strong> <br> <br>
You need to solve the problem located at <a href=/problem?pretty=true>/problem</a>. <br>
There is a labyrinth from which you have to escape. <br> <br>
If you manage to escape from the labyrinth, you get the flag. <br> <br>
You spawn in this labyrinth, in which there are monsters (which are bots controlled by the server), walls and an escape point. Sometimes you just can't escape...<br>
Every time you make an action (even no_action), monsters make an action; you can't go throughout walls, and if you come across a monster, you lose. <br> <br>
You first make a GET request to have an instance of the problem as a session, JSON encoded. You can also do the GET request with parameter pretty=true to better visualize the labyrinth. With other GET requests, you see your session data as JSON (and you still have the 'pretty' option). <br>
Then you POST with parameter 'action', possible values: move_left, move_up, move_right, move_down, no_action. The response is a JSON containing the actions computed by the server, i.e. the action you specified if it was legal and the monsters' actions; if you come across a monster, you get the message "Wasted!", and if you manage to arrive to the escape cell, you get the flag.<br>
Monsters are labeled as monster_1 ... monster_N according to the order in which they are encountered iterating on cells from left to right and then down with carriage return, i.e. cell through columns through rows. <br>
You can't solve the problem 'by hand' because session expires after 2 seconds, so you only have 2 seconds for each move. <br>
After session expires, you are redirected to another session. <br>

