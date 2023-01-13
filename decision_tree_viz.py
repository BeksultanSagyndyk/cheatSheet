from sklearn.tree import export_graphviz
# Export as dot file
# clf - модель, X_train_b2b.columns - название признаков
export_graphviz(clf, out_file='tree.dot',
                feature_names = X_train_b2b.columns,
                class_names = ['0','1'],
                rounded = True, proportion = False,
                precision = 2, filled = True)
 
# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
 
# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')
