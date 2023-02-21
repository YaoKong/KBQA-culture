import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randn(6, 4),  columns=list('ABCD'))
tmp = pd.DataFrame([],  columns=list('ABCD'))

print(df.loc[0]["A"])

