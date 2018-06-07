import pandas as pd
import sys
filename = sys.argv[1]
idxs = pd.read_csv(filename, index_col=None, header=None).values.T[0]
print(*idxs)