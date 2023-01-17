import matplotlib.pyplot as plt
plt.figure(figsize=(12,9))
plt.pie(piechart['label'],labels=piechart['index'],
        startangle=90,
        shadow=True,
        explode=(0.2, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1, 0.1, 0.1, 0.1),
        autopct='%1.2f%%')
