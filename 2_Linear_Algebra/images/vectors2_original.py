import matplotlib.pyplot as plt

vectors = [
	{
		'values': [0.6, 0.8],
		'color': '#27ae60',
	},
	{
		'values': [0.8, -0.6],
		'color': '#2980b9',
	},
	{
		'values': [1, 0],
		'color': '#f39c12',
	},
	{
		'values': [0, 1],
		'color': '#c0392b',
	},
]

box = [
	{
		'x': [1.4, -0.2],
		'y': [0.2, 1.4],
		'color': '#34495e',
	},
	{
		'x': [-0.2, -1.4],
		'y': [1.4, -0.2],
		'color': '#34495e',
	},
	{
		'x': [-1.4, 0.2],
		'y': [-0.2, -1.4],
		'color': '#34495e',
	},
	{
		'x': [0.2, 1.4],
		'y': [-1.4, 0.2],
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

ax.set_ylim(-2, 2)
ax.set_xlim(-2, 2)
ax.set_aspect('equal')
fig.savefig('vectors2_original.png', bbox_inches='tight', dpi=72)
