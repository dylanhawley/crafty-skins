from skinpy import Skin, Perspective

skin = Skin.from_path("steve.png")

# create a perspective from which to view the render
perspective = Perspective(
  x="left",
  y="front",
  z="up",
  scaling_factor=5, # bigger numbers mean bigger image
)

# save the render
skin.to_isometric_image(perspective).save("render.png")