import yaml
from engine import Engine
from animation import Animation
import matplotlib; matplotlib.use('TkAgg')

# -----------------------------------
# Load configuration
# -----------------------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# -----------------------------------
# Create engine and animation
# -----------------------------------
engine = Engine(cfg)
anim = Animation(engine)

# -----------------------------------
# Run
# -----------------------------------
anim.run()
