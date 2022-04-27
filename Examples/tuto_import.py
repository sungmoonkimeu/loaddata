import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

print(os.path.dirname(os.path.dirname(__file__))+'\My_library')
sys.path.append(os.path.dirname(os.path.dirname(__file__))+'\My_library')

import draw_poincare_plotly as PS

print(os.getcwd())
#os.chdir('loaddata\Examples')
fn = 'LP0_edited.txt'
data = pd.read_table(fn, delimiter=r"\s+")
time = pd.to_numeric(data['Index']) / 10000
S0 = pd.to_numeric(data['S0(mW)'])
S1 = pd.to_numeric(data['S1'])
S2 = pd.to_numeric(data['S2'])
S3 = pd.to_numeric(data['S3'])

fig = PS.PS5()

cm = np.linspace(0, 1, len(S1))  # color map
cm[-1] = 1.3
#fig.add_scatter3d(data, x='S1', y='S2', z='S3', mode="markers", marker=dict(size=1))
fig.add_scatter3d(x=S1, y=S2, z=S3, mode="markers",
                    marker=dict(size=3, color=cm, colorscale='amp'), name='F1')
fig.show()


fig = px.colors.sequential.swatches_continuous()
fig.show()
fig = px.colors.cyclical.swatches_cyclical()
fig.show()