import os
import sys

root_dir = os.path.join(os.getcwd(), "src")
if not root_dir in sys.path:
    sys.path.insert(1, root_dir)

