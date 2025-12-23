import numpy as np
from pathlib import Path


GEOMETRY_PATH = Path("/baai-cwm-backup/cwm/tong.liu/Geo_Out_Fine")
FILENAME = "004125.npz"


geometry_path = GEOMETRY_PATH / FILENAME

geometry = np.load(geometry_path)
point_map, mask = geometry['point_map'], geometry['mask']

print(point_map.shape)
print(mask.shape)

