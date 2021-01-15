import torch


def episodic_collate_fn(input_data):
    """
    Collate function to be used as argument for the collate_fn parameter of episodic data loaders.
    Args:
        input_data (list[tuple(Tensor, int, int)]): each element is a tuple containing:
            - an image as a torch Tensor
            - the label of this image
            - the group of this image

    Returns:
        tuple(Tensor, Tensor, Tensor, Tensor, list[int], int, int): respectively:
            - source images,
            - their labels,
            - target images,
            - their labels,
            - the dataset class ids of the class sampled in the episode
            - source domain
            - target domain
    """
    source = input_data[0][2]
    target = input_data[-1][2]
    assert all(item[2] == source or item[2] == target for item in input_data)

    true_class_ids = list(set([x[1] for x in input_data]))

    images_source = torch.cat([x[0].unsqueeze(0) for x in input_data if x[2] == source])
    labels_source = torch.tensor(
        [true_class_ids.index(x[1]) for x in input_data if x[2] == source]
    )

    images_target = torch.cat([x[0].unsqueeze(0) for x in input_data if x[2] == target])
    labels_target = torch.tensor(
        [true_class_ids.index(x[1]) for x in input_data if x[2] == target]
    )

    return (
        images_source,
        labels_source,
        images_target,
        labels_target,
        true_class_ids,
        int(source),
        int(target),
    )
