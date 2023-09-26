# tmux sessions

Create a new session:

\```
tmux new-session -s "name"
\```

How to enter a session:

\```
tmux attach -t "name"
\```

How to kill a session:

\```
tmux kill-session -t "name"
\```

# How to clear swap and cache on the server

Clear the swap:

\```
sudo swapoff -a && sudo swapon -a
\```

Clear the cache:

\```
sudo mc -e /proc/sys/vm/drop_caches ---> Enter the value 3 in the opened window
\```

# How to create a new environment

Create a new Python environment:

\```
python3 -m venv myenv
\```

Activate the environment:

\```
source myenv/bin/activate
\```

To add the environment to kernels in Jupyter Notebook:

\```
pip install ipykernel
ipython kernel install --user --name=envname
\```

# CrossJoin in Pandas

\```
s['key'] = 0
ss['key'] = 0
res = ss.merge(s, on='key', how='outer')
\```

# Run a long-playing script in tmux

\```
nohup python blip-large-training-inference.py > prints_debug.out 2>&1 &
\```

# Environment

Create a new conda environment:

\```
conda create -n envname python=3.10
conda activate envname
pip install ipykernel
/path/to/envname/bin/python -m ipykernel install --user --name=envname
\```

To delete a specific environment:

\```
conda env list
conda deactivate
conda remove --name ENV_NAME --all
\```

The `--all` flag removes all the packages installed in that environment.
