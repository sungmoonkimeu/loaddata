import numpy as np
import plotly.graph_objects as go

x = np.linspace(-1,1, 50)
y = np.linspace(-1,3, 100)
x, y = np.meshgrid(x,y)

z = x**3-3*y*x+1  # the surface eqn

fig = go.Figure()
fig.add_surface(x=x, y=y, z=z, colorscale='Reds_r', colorbar_thickness=25, colorbar_len=0.75, opacity=0.5);

X = np.linspace(-1,1, 6)
Y = np.linspace(-1, 3, 12)
#Define the first family of coordinate lines
X, Y = np.meshgrid(X,Y)
Z = X**3-3*Y*X+1
line_marker = dict(color='#999914', width=4)
for xx, yy, zz in zip(X, Y, Z+0.03):
    fig.add_scatter3d(x=xx, y=yy, z=zz, mode='lines', line=line_marker, name='')
#Define the second family of coordinate lines
Y, X = np.meshgrid(Y, X)
Z = X**3-3*Y*X+1
line_marker = dict(color='#101010', width=4)
for xx, yy, zz in zip(X, Y, Z+0.03):
    fig.add_scatter3d(x=xx, y=yy, z=zz, mode='lines', line=line_marker, name='')
fig.update_layout(width=700, height=700, showlegend=False)
fig.show()