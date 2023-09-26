# tmux sessions

Create a new session:

```bash
tmux new-session -s "name"
```

How to enter a session:

```bash
tmux attach -t "name"
```

How to kill a session:

```bash
tmux kill-session -t "name"
```

# How to clear swap and cache on the server

Clear the swap:

```bash
sudo swapoff -a && sudo swapon -a
```

Clear the cache:

```bash
sudo mc -e /proc/sys/vm/drop_caches ---> Enter the value 3 in the opened window
```

# How to create a new environment

Create a new Python environment:

```bash
python3 -m venv myenv
```

Activate the environment:

```bash
source myenv/bin/activate
```

To add the environment to kernels in Jupyter Notebook:

```bash
pip install ipykernel
ipython kernel install --user --name=envname
```

# CrossJoin in Pandas

```python
s['key'] = 0
ss['key'] = 0
res = ss.merge(s, on='key', how='outer')
```

# Run a long-playing script in tmux

```bash
nohup python blip-large-training-inference.py > prints_debug.out 2>&1 &
```

# Environment

Create a new conda environment:

```bash
conda create -n envname python=3.10
conda activate envname
pip install ipykernel
/path/to/envname/bin/python -m ipykernel install --user --name=envname
```

To delete a specific environment:

```bash
conda env list
conda deactivate
conda remove --name ENV_NAME --all
```

The `--all` flag removes all the packages installed in that environment.
