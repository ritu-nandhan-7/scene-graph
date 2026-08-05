from loader import VisualGenomeLoader
from visualisations import *
loader = VisualGenomeLoader(".")

scene = loader.load_scene(0)

show_scene(scene,num_objects=5,num_rels=0)