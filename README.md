# tmux sessions
tmux new-session -s "name"
 
How to enter session 
tmux attach -t "name"
 
how to kill
tmux kill-session -t "name"

# how to clear swap and cache on server
sudo swapoff -a && sudo swapon -a  (очистка swap)

sudo mc -e /proc/sys/vm/drop_caches ---> в открывшемся окне записать значение 3 (очистка cache)

# how to create new env
python3 -m venv myenv
# how to activate it
source myenv/bin/activate
# how to add it to kernels in jupyter notebook
pip install ipykernel

ipython kernel install --user --name=envname 
