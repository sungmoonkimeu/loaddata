import os
import sys
import pandas as pd
import plotly.graph_objects as go

print(os.path.dirname(os.path.dirname(__file__))+'\My_library')
sys.path.append(os.path.dirname(os.path.dirname(__file__))+'\My_library')

import draw_poincare_plotly as PS

print(os.getcwd())
os.chdir('loaddata\Examples')
fn = 'LP0_edited.txt'
data = pd.read_table(fn, delimiter=r"\s+")
time = pd.to_numeric(data['Index']) / 10000
S0 = pd.to_numeric(data['S0(mW)'])
S1 = pd.to_numeric(data['S1'])
S2 = pd.to_numeric(data['S2'])
S3 = pd.to_numeric(data['S3'])

fig = PS.PS5()
#fig.add_scatter3d(data, x='S1', y='S2', z='S3', mode="markers", marker=dict(size=1))
fig.add_scatter3d(x=S1, y=S2, z=S3, mode="markers", marker=dict(size=1))
fig.show()