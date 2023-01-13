# Старт новой сессии со своим именем
tmux new-session -s "name"
 
 
Заходим в существующую сессию
tmux attach -t "name"
 
Отключить сессию
tmux kill-session -t "name"

# Очистка swap and cache на сервере
sudo swapoff -a && sudo swapon -a  (очистка swap)

sudo mc -e /proc/sys/vm/drop_caches ---> в открывшемся окне записать значение 3 (очистка cache)
