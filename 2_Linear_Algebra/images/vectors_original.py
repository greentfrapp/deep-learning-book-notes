import matplotlib.pyplot as plt

vectors = [
	{
		'values': [1, 0],
		'color': '#27ae60',
	},
	{
		'values': [0, 1],
		'color': '#2980b9',
	},
	{
		'values': [1, 1],
		'color': '#f39c12',
	},
	{
		'values': [-1, -2],
		'color': '#c0392b',
	},
]

box = [
	{
		'x': [-1, 1],
		'y': [1, 1],
		'color': '#34495e',
	},
	{
		'x': [1, 1],
		'y': [1, -1],
		'color': '#34495e',
	},
	{
		'x': [1, -1],
		'y': [-1, -1],
		'color': '#34495e',
	},
	{
		'x': [-1, -1],
		'y': [-1, 1],
		'color': '#34495e',
	},
]

fig, ax = plt.subplots()

for vector in vectors:
	ax.arrow(
		x=0,
		y=0,
		dx=vector['values'][0],
		dy=vector['values'][1],
		head_width=0.1,
		head_length=0.1,
		color=vector['color'],
	)

for side in box:
	ax.plot(
		side['x'],
		side['y'],
		color=side['color'],
		linestyle='--',
		linewidth=0.5,
	)

ax.set_ylim(-5, 5)
ax.set_xlim(-5, 5)
ax.set_aspect('equal')
fig.savefig('vectors_original.png', bbox_inches='tight', dpi=72)
