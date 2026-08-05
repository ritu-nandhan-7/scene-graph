import matplotlib.pyplot as plt
import matplotlib.patches as patches

def print_image_info(scene):

    print("IMAGE INFORMATION")
    print("-" * 70)

    print(f"Image Index : {scene.info.image_index}")
    print(f"Image ID    : {scene.info.image_id}")
    print(f"Resolution  : {scene.info.width} × {scene.info.height}")
    print(f"Objects     : {len(scene.objects)}")
    print(f"Relations   : {len(scene.relationships)}")


def print_objects(scene,limit=None):

    print()
    print("OBJECTS")
    print("-" * 70)

    objects = scene.objects if limit is None else scene.objects[:limit]

    for i, obj in enumerate(objects):

        print(f"[{i}] {obj.label}")

        print(
            f"Bounding Box : "
            f"{obj.bbox.as_xywh()}"
        )

        print()

def print_relationships(scene,limit=None):

    print()
    print("RELATIONSHIPS")
    print("-" * 70)

    relationships = (scene.relationships if limit is None else scene.relationships[:limit])
    
    for rel in relationships:

        print(
            f"{rel.subject.label}"
            f" ---- {rel.predicate} ----> "
            f"{rel.object.label}"
        )

def draw_scene(
    scene,
    num_objects=None,
    show_labels=True,
    figsize=(12, 8)
):
    """
    Draws all object bounding boxes on an image.

    Parameters
    ----------
    scene : Scene

    num_objects : int | None
        Draw only first N objects.
        None -> draw all.

    show_labels : bool
        Display object labels.

    figsize : tuple
        Matplotlib figure size.
    """

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(scene.image)

    colors = [
        "red",
        "lime",
        "cyan",
        "yellow",
        "magenta",
        "orange",
        "blue",
        "white",
    ]

    objects = scene.objects

    if num_objects is not None:
        objects = objects[:num_objects]

    for i, obj in enumerate(objects):

        color = colors[i % len(colors)]

        x = obj.bbox.x1
        y = obj.bbox.y1
        w = obj.bbox.width
        h = obj.bbox.height

        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor=color,
            linewidth=2
        )

        ax.add_patch(rect)

        if show_labels:

            ax.text(
                x,
                y - 5,
                f"{obj.label} ({obj.object_index})",
                fontsize=9,
                color="white",
                bbox=dict(
                    facecolor=color,
                    alpha=0.7,
                    edgecolor="none",
                    pad=2
                )
            )

    ax.set_title(
        f"Image {scene.info.image_id}"
    )

    ax.axis("off")

    plt.tight_layout()

    plt.show()

def show_scene(
    scene,
    num_objects=None,
    num_rels=None,
    show_labels=True
):
    """
    Prints the scene and visualizes it.
    """

    print_image_info(scene)

    print_objects(scene,limit=num_objects)

    print_relationships(scene,limit=num_rels)

    draw_scene(
        scene,
        num_objects=num_objects,
        show_labels=show_labels
    )

