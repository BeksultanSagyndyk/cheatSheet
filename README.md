# cheatSheet
#Старт новой сессии со своим именем
tmux new-session -s "name"
 
 
#Заходим в существующую сессию
tmux attach -t "name"
 
#Отключить сессию
tmux kill-session -t "name"
