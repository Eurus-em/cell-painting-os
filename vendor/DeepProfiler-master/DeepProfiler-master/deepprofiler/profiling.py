"""EfficientNet-based feature extraction pipeline for microscopy images."""

import os

import efficientnet.tfkeras as efn
import numpy as np
import tensorflow as tf

from deepprofiler.dataset.utils import tic, toc
from deepprofiler.imaging.boxes import get_locations, prepare_boxes
from deepprofiler.imaging.cropping import SingleImageCropGenerator

# Module-level cache of already-created output directories.
# Images from the same plate/well share a directory; caching avoids a
# redundant os.makedirs system call for every image after the first.
_seen_dirs = set()

tf.compat.v1.disable_v2_behavior()
tf.config.run_functions_eagerly(False)

_EFFICIENTNET_MODELS = {
    0: efn.EfficientNetB0,
    1: efn.EfficientNetB1,
    2: efn.EfficientNetB2,
    3: efn.EfficientNetB3,
    4: efn.EfficientNetB4,
    5: efn.EfficientNetB5,
    6: efn.EfficientNetB6,
    7: efn.EfficientNetB7,
}


def build_model(config):
    """Build the EfficientNet feature extraction model from config.

    In standard profiling mode the model is built with the number of channels
    specified in the dataset config (e.g. 5 for Cell Painting) and no ImageNet
    weights.  The final layers are GlobalAveragePooling2D (named ``pool5``) and
    a softmax Dense head (named ``ClassProb``).  Weights are loaded separately
    via :meth:`Profile.configure`.

    If ``config["profile"]["use_pretrained_input_size"]`` is set, builds a
    3-channel ImageNet-pretrained model at that spatial resolution instead —
    useful for fine-tuning experiments.
    """
    num = config["train"]["model"]["params"]["conv_blocks"]
    if num not in _EFFICIENTNET_MODELS:
        raise ValueError(f"{num} not in supported EfficientNet variants: {list(_EFFICIENTNET_MODELS)}")

    if config["profile"].get("use_pretrained_input_size"):
        size = config["profile"]["use_pretrained_input_size"]
        inp = tf.keras.layers.Input((size, size, 3), name="input")
        return _EFFICIENTNET_MODELS[num](input_tensor=inp, include_top=True, weights="imagenet")

    h = w = config["dataset"]["locations"]["box_size"]
    c = len(config["dataset"]["images"]["channels"])
    inp = tf.keras.layers.Input((h, w, c))
    base = _EFFICIENTNET_MODELS[num](input_tensor=inp, include_top=False, weights=None)
    features = tf.keras.layers.GlobalAveragePooling2D(name="pool5")(base.layers[-1].output)
    y = tf.keras.layers.Dense(config["num_classes"], activation="softmax", name="ClassProb")(features)
    return tf.keras.Model(inputs=inp, outputs=[y])


class Profile(object):
    """Extract per-cell deep learning features from microscopy images.

    This class implements the full profiling pipeline:

    1. Build an EfficientNet model matching the dataset channel count.
    2. Load pre-trained weights from a checkpoint (e.g. Cell Painting CNN v1).
    3. For each image, crop patches around cell centroids and run them through
       the network, collecting the activations at a named intermediate layer.
    4. Write per-image ``.npz`` files containing a ``features`` array of shape
       ``(num_cells, feature_dim)``.

    Typical usage::

        dset = deepprofiler.dataset.image_dataset.read_dataset(config)
        deepprofiler.profiling.profile(config, dset)

    Or step-by-step (e.g. in tests)::

        prof = Profile(config, dset)
        prof.configure()
        prof.extract_features(key, image_array, meta)
    """

    def __init__(self, config, dset):
        """Initialise the profiler and build the model graph.

        Args:
            config: Experiment configuration dict.  Must include
                ``dataset``, ``train``, ``profile``, ``paths``, and
                ``num_classes`` keys.
            dset: :class:`~deepprofiler.dataset.image_dataset.ImageDataset`
                with at least one target added via ``add_target``.
        """
        self.config = config
        self.dset = dset
        self.num_channels = len(self.config["dataset"]["images"]["channels"])

        self.config["num_classes"] = self.dset.targets[0].shape[1]
        self.feature_model = build_model(self.config)
        self.profile_crop_generator = SingleImageCropGenerator(config, dset)

    def configure(self):
        """Start the crop generator and load checkpoint weights.

        Loads weights from ``config["paths"]["checkpoints"] / config["profile"]["checkpoint"]``.
        The classifier head is always renamed before loading so that the
        checkpoint's Dense head is skipped and only backbone weights are loaded.

        After loading, builds ``self.feat_extractor``: a sub-model whose output
        is the activation of ``config["profile"]["feature_layer"]`` (e.g.
        ``"pool5"`` for the 1280-d GlobalAveragePooling2D embedding).

        Also builds ``self.combined_crop_features``, which connects the crop
        graph output directly to the feature extractor so that cropping and
        inference run in a single ``session.run`` call — avoiding an extra GPU
        kernel launch and CPU-GPU round-trip per image.
        """
        self.profile_crop_generator.start(tf.compat.v1.keras.backend.get_session())

        if self.config["profile"]["checkpoint"] != "None":
            checkpoint = self.config["paths"]["checkpoints"] + "/" + self.config["profile"]["checkpoint"]
            self.feature_model.layers[-1]._name = "classifier"
            self.feature_model.load_weights(checkpoint, by_name=True)

        self.feature_model.summary()
        self.feat_extractor = tf.compat.v1.keras.Model(
            self.feature_model.inputs,
            self.feature_model.get_layer(self.config["profile"]["feature_layer"]).output
        )
        print("Extracting output from layer:", self.config["profile"]["feature_layer"])

        # Build the combined crop→features graph.  The crop generator already
        # holds placeholders and the crop op in self.input_variables; we wire
        # the crop output directly into the feature extractor so that a single
        # session.run produces features without an intermediate CPU round-trip.
        crop_output = self.profile_crop_generator.input_variables["labeled_crops"][0]
        self.combined_crop_features = self.feat_extractor(crop_output)

    def check(self, meta):
        """Return True if this image still needs to be profiled.

        Skips images whose output ``.npz`` already exists, enabling resumable
        profiling runs.

        Args:
            meta: A row from the metadata DataFrame (Pandas Series or dict-like)
                with ``Metadata_Plate``, ``Metadata_Well``, and ``Metadata_Site``.
        """
        output_file = self.config["paths"]["features"] + "/{}/{}/{}.npz"
        output_file = output_file.format(meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])
        if os.path.isfile(output_file):
            print("Already done:", output_file)
            return False
        return True

    def extract_features(self, key, image_array, meta):
        """Extract and save features for a single image.

        Builds a feed dict for the crop graph, then runs a **single**
        ``session.run`` that performs both cropping and feature extraction
        in one fused GPU call — avoiding the intermediate CPU round-trip
        and second kernel launch that the original two-call design incurred.

        Output path: ``{paths.features}/{Plate}/{Well}/{Site}.npz``

        The ``.npz`` file contains:
        - ``features``: float array of shape ``(num_cells, feature_dim)``
        - ``metadata``: dict of metadata fields plus ``Metadata_Model``
        - ``locations``: cell centroid coordinates from the locations CSV

        Args:
            key: Image key (index into the dataset); may be ``None`` when
                called directly.
            image_array: numpy array of shape ``(H, W, C)`` where C matches
                the configured channel count.
            meta: Metadata row (Pandas Series) for this image.
        """
        start = tic()
        output_file = self.config["paths"]["features"] + "/{}/{}/{}.npz"
        output_file = output_file.format(meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])
        out_dir = self.config["paths"]["features"] + "/{}/{}".format(meta["Metadata_Plate"], meta["Metadata_Well"])
        if out_dir not in _seen_dirs:
            os.makedirs(out_dir, exist_ok=True)
            _seen_dirs.add(out_dir)

        batch_size = self.config["profile"]["batch_size"]
        image_key, image_names, outlines = self.dset.get_image_paths(meta)

        # --- Build the feed dict (inlined from SingleImageCropGenerator.prepare_image) ---
        num_targets = len(self.dset.targets)
        crop_locations = get_locations(image_key, self.config, random_sample=None)
        total_crops = len(crop_locations)
        if total_crops == 0:
            print("No cells to profile:", output_file)
            return

        if (self.config["dataset"]["images"]["width"] != image_array.shape[1] or
                self.config["dataset"]["images"]["height"] != image_array.shape[0]):
            config_shape = (self.config["dataset"]["images"]["width"], self.config["dataset"]["images"]["height"])
            im_shape = (image_array.shape[1], image_array.shape[0])
            raise ValueError("Loaded image shape WxH " + str(im_shape) +
                             " != configured image shape WxH " + str(config_shape))

        # Prepare boxes, targets, and batch the same way the crop generator does.
        batch = {"images": [], "locations": [], "targets": [[] for _ in range(num_targets)]}
        batch["images"].append(image_array)
        batch["locations"].append(crop_locations)
        for i in range(num_targets):
            batch["targets"][i].append(self.dset.targets[i].get_values(meta))

        boxes, box_ind, targets, mask_ind = prepare_boxes(batch, self.config)
        feed_dict = {
            self.profile_crop_generator.input_variables["image_ph"]: image_array[np.newaxis, ...],
            self.profile_crop_generator.input_variables["boxes_ph"]: boxes,
            self.profile_crop_generator.input_variables["box_ind_ph"]: box_ind,
            self.profile_crop_generator.input_variables["mask_ind_ph"]: mask_ind,
        }
        for i in range(num_targets):
            tname = "target_" + str(i)
            feed_dict[self.profile_crop_generator.input_variables["targets_phs"][tname]] = targets[i]

        # --- Single fused call: crop + feature extraction in one GPU launch ---
        session = tf.compat.v1.keras.backend.get_session()
        feats = session.run(self.combined_crop_features, feed_dict)

        while len(feats.shape) > 2:
            feats = np.mean(feats, axis=1)

        key_values = {k: meta[k] for k in meta.keys()}
        key_values["Metadata_Model"] = "efficientnet"
        # Use np.savez (no compression) — float feature arrays compress poorly
        # with DEFLATE, so the CPU overhead of compression isn't worth the
        # marginal (10-20%) disk savings for large-scale profiling runs.
        np.savez(output_file, features=feats, metadata=key_values, locations=crop_locations)
        toc(image_key + " (" + str(total_crops) + " cells)", start)


def profile(config, dset):
    """Run the full profiling pipeline: configure, scan all images, write .npz files."""
    p = Profile(config, dset)
    p.configure()
    dset.scan(p.extract_features, frame="all", check=p.check)
    print("Profiling: done")
