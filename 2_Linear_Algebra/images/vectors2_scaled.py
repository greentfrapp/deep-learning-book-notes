import matplotlib.pyplot as plt

vectors = [
	{
		'values': [0.9, 1.2],
		'color': '#27ae60',
	},
	{
		'values': [0.4, -0.3],
		'color': '#2980b9',
	},
	{
		'values': [0.86, 0.48],
		'color': '#f39c12',
	},
	{
		'values': [0.48, 1.14],
		'color': '#c0392b',
	},
]

box = [
	{
		'x': [1.3, 0.5],
		'y': [0.9, 1.5],
		'color': '#34495e',
	},
	{
		'x': [0.5, -1.3],
		'y': [1.5, -0.9],
		'color': '#34495e',
	},
	{
		'x': [-1.3, -0.5],
		'y': [-0.9, -1.5],
		'color': '#34495e',
	},
	{
		'x': [-0.5, 1.3],
		'y': [-1.5, 0.9],
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
fig.savefig('vectors2_scaled.png', bbox_inches='tight', dpi=72)
