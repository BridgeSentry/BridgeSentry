import os
from datetime import datetime

from config.constants import Bridge
from dataset_generator.generator import GraphDatasetGenerator


def build_dataset_group_name(dataset_type: str, new_dataset_value: str | None) -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    tags = [tag.strip() for tag in (new_dataset_value or "").split(",") if tag.strip()]
    return "-".join([date_str, dataset_type, *tags])


def dataset_dir_is_complete(dataset_dir: str) -> bool:
    """Whether a previously-generated PyG dataset exists at dataset_dir.

    Checks the two artifacts every processed dataset always has
    (data_index.csv, metadata.json), not just directory existence —
    GraphDatasetGenerator and PyG's InMemoryDataset both eagerly
    os.makedirs() raw/processed as a side effect of construction, so a
    bare os.path.isdir() check would wrongly call an interrupted/partial
    generation "complete".
    """
    processed_dir = os.path.join(dataset_dir, "processed")
    return (
        os.path.isfile(os.path.join(processed_dir, "data_index.csv"))
        and os.path.isfile(os.path.join(processed_dir, "metadata.json"))
    )


def resolve_dataset_and_reports_root(
    dataset_type: str,
    load_data: str | None,
    new_dataset_tags: str | None,
    force_reload: bool,
) -> tuple[str, str, bool]:
    """Resolve (dataset_path, reports_root, resolved_force_reload) for a training run.

    - load_data set: no generation; reports_root is the parent dir of the
      given path, so reports land beside whatever dataset folder the user
      pointed at. force_reload passes through unchanged.
    - new_dataset_tags set (load_data is None): builds
      outputs/YYYYMMDD-datasettype-tags/, with _dataset/ as the dataset
      root and the group dir itself as reports_root. Generates the dataset
      if missing, warns-and-reuses if present (unless force_reload),
      regenerates if force_reload is set.
    """
    if load_data:
        dataset_path = load_data
        reports_root = os.path.dirname(os.path.normpath(dataset_path))
        return dataset_path, reports_root, force_reload

    group_name = build_dataset_group_name(dataset_type, new_dataset_tags)
    group_dir = os.path.abspath(os.path.join("outputs", group_name))
    dataset_path = os.path.join(group_dir, "_dataset")

    if dataset_dir_is_complete(dataset_path) and not force_reload:
        print(
            f"[warn] Dataset already exists at '{dataset_path}'; reusing it "
            f"(pass --force-reload to regenerate)."
        )
        return dataset_path, group_dir, False

    generator = GraphDatasetGenerator(
        bridges=list(Bridge),
        dataset_type=dataset_type,
        output_folder=dataset_path,
        force_reload=force_reload,
    )
    generator.generate_graph_dataset()
    return dataset_path, group_dir, False
