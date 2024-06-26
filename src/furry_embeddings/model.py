import os
import tensorflow as tf
from keras import Model
import numpy as np
import typing

from sklearn.preprocessing import MultiLabelBinarizer


action_tags = [
    "age_difference",
    "anal",
    "bdsm",
    "bite",
    "blood",
    "blush",
    "bound",
    "breath",
    "cum",
    "dialogue",
    "dominant",
    "drinking",
    "dripping",
    "ejaculation",
    "embrace",
    "exercise",
    "eyes_closed",
    "fart",
    "feces",
    "fingering",
    "first_person_view",
    "forced",
    "gore",
    "lactating",
    "leaking",
    "leaning",
    "leg_grab",
    "looking_at_another",
    "looking_at_genitalia",
    "looking_at_viewer",
    "looking_back",
    "looking_pleasured",
    "lying",
    "masturbation",
    "messy",
    "mind_control",
    "musk",
    "on_bottom",
    "on_top",
    "one_eye_closed",
    "open_mouth",
    "oral",
    "orgasm",
    "penetration",
    "penile",
    "precum",
    "presenting",
    "pussy_juice",
    "roleplay",
    "saliva",
    "seductive",
    "sex",
    "sitting",
    "sitting_on_another",
    "size_difference",
    "smile",
    "smiling_at_viewer",
    "sniffing",
    "spreading",
    "squish",
    "standing",
    "submissive",
    "substance_intoxication",
    "sweat",
    "tears",
    "transformation",
    "undressing",
    "urine",
    "vaginal",
    "vore",
    "wet",
]
body_tags = [
    "anatomically_correct",
    "animal_genitalia",
    "anthro",
    "anus",
    "areola",
    "balls",
    "beak",
    "belly",
    "body_hair",
    "bone",
    "breasts",
    "bulge",
    "butt",
    "claws",
    "clothed",
    "curvy_figure",
    "disembodied_hand",
    "disembodied_penis",
    "erection",
    "eyebrows",
    "eyelashes",
    "fangs",
    "feathers",
    "feet",
    "feral",
    "fin",
    "fingers",
    "fur",
    "hair",
    "hooves",
    "horn",
    "hyper",
    "knot",
    "lips",
    "macro",
    "mane",
    "markings",
    "micro",
    "mostly_nude",
    "muscular",
    "narrowed_eyes",
    "navel",
    "nipples",
    "not_furry",
    "nude",
    "overweight",
    "pawpads",
    "paws",
    "penis",
    "piercing",
    "plant",
    "pupils",
    "pussy",
    "scales",
    "scar",
    "slit",
    "spots",
    "stripes",
    "tail",
    "teeth",
    "tentacles",
    "tongue",
    "tuft",
    "unusual_anatomy",
    "wide_hips",
    "wings",
]
clothing_tags = [
    "accessory",
    "apron",
    "armor",
    "armwear",
    "bottomwear",
    "cape",
    "chastity_device",
    "collar",
    "costume",
    "diaper",
    "dress",
    "eyewear",
    "footwear",
    "gag",
    "handwear",
    "harness",
    "headgear",
    "headphones",
    "jewelry",
    "legwear",
    "lingerie",
    "mask",
    "restraints",
    "ribbons",
    "ring",
    "robe",
    "rope",
    "rubber",
    "sex_toy",
    "suit",
    "swimwear",
    "topwear",
    "underwear",
    "uniform",
    "weapon",
]
identity_tags = [
    "ambiguous_gender",
    "crossgender",
    "female",
    "intersex",
    "male",
    "young",
]
species_tags = [
    "ailurid",
    "alien",
    "amphibian",
    "animate_inanimate",
    "arthropod",
    "bat",
    "bear",
    "bird",
    "bovine",
    "caprine",
    "cephalopod",
    "cetacean",
    "crocodilian",
    "deer",
    "demon",
    "digimon_(species)",
    "dinosaur",
    "domestic_dog",
    "dragon",
    "elemental_creature",
    "equid",
    "eulipotyphlan",
    "feline",
    "fish",
    "flora_fauna",
    "fox",
    "goo_creature",
    "human",
    "humanoid",
    "hyena",
    "kobold",
    "lagomorph",
    "lizard",
    "marsupial",
    "mephitid",
    "mollusk",
    "monster",
    "mustelid",
    "mythological_avian",
    "pantherine",
    "pokemon_(species)",
    "primate",
    "procyonid",
    "robot",
    "rodent",
    "snake",
    "spirit",
    "suina",
    "taur",
    "wolf",
]


class EmbeddingModel:
    def __init__(
        self,
        model_path: typing.Union[str, os.PathLike],
    ) -> None:
        self.full_model = tf.keras.models.load_model(model_path)
        input_layer = self.full_model.layers[0]

        hidden_layer = self.full_model.get_layer("feature_layer")
        self.feature_model = tf.keras.Model(
            inputs=input_layer.input, outputs=hidden_layer.output
        )

        self.tf_config = self.full_model.get_config()
        self.img_size = self.tf_config["layers"][0]["config"]["batch_input_shape"][1]
        self.channels = self.tf_config["layers"][0]["config"]["batch_input_shape"][3]
        self.layer_order = [
            layer[0][: layer[0].find("_")] if layer[0].find("_") != -1 else layer[0]
            for layer in self.tf_config["output_layers"]
        ]

        self.normalized = [(0.5, 0.5, 0.5), (0.225, 0.225, 0.225)]

        # load multilabel binarizers for model
        self.mlbs = {
            "action": MultiLabelBinarizer().fit(action_tags),
            "body": MultiLabelBinarizer().fit(body_tags),
            "clothing": MultiLabelBinarizer().fit(clothing_tags),
            "identity": MultiLabelBinarizer().fit(identity_tags),
            "species": MultiLabelBinarizer().fit(species_tags),
        }
        self.tags = {k: v.classes_ for k, v in self.mlbs.items()}

    def _load_image(self, image: np.ndarray) -> tf.Tensor:
        """Turns image from a numpy array into a properly formatted tensor:
        converting to RGB, resizing to fit the model, and Normalizing the pixel values
        between [0.0, 1.0]

        Args:
            img_bytes np.ndarray: Image numpy array

        Returns:
            Tensor: Tensor representation of the Image
        """
        # Resize Image
        image_tensor = tf.convert_to_tensor(image)
        image_resized = tf.image.resize(image_tensor, [self.img_size, self.img_size])

        # Normalize it from [0, 255] to [0.0, 1.0]
        image_normalized = image_resized / 255.0
        return image_normalized

    def _normalize_func(self, image):
        img_mean = self.normalized[0]
        img_stddev = self.normalized[1]

        offset = tf.constant(img_mean, shape=[1, 1, 3])
        image -= offset
        scale = tf.constant(img_stddev, shape=[1, 1, 3])
        image /= scale
        return image

    def _get_rating(self, value: float):
        if value >= 2 / 3:
            return "explicit"
        elif value >= 1 / 3:
            return "questionable"
        else:
            return "safe"

    def _get_dataset(self, *images):
        ds = tf.data.Dataset.from_tensor_slices([img for img in images])
        ds = ds.map(self._normalize_func, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(8, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def image_latent_vector(
        self,
        *images: np.ndarray,
    ) -> np.ndarray:
        """TODO: Make summary

        Raises:
            ValueError: _description_

        Returns:
            np.ndarray: _description_
        """
        if len(images) == 0:
            raise ValueError("There must be at least one image given.")

        imgs = [self._load_image(image) for image in images]
        ds = self._get_dataset(*imgs)
        self.feature_model: Model
        res = self.feature_model.predict(ds)
        return np.array(res)

    def predict_image_tags(self, *images, t: float = 0.5):
        if len(images) == 0:
            raise ValueError("There must be at least one image given.")

        imgs = [self._load_image(image) for image in images]
        ds = self._get_dataset(*imgs)
        x = self.full_model.predict(ds)

        num_images = len(images)
        res = [[] for _ in range(num_images)]
        for layer, result in zip(self.layer_order, x):

            if layer == "rating":
                out = [
                    [
                        {
                            "tag": self._get_rating(r[0]),
                            "value": float(r[0]),
                            "category": layer,
                        }
                    ]
                    for r in result
                ]

            else:
                result = np.array(result)
                result_mask = np.where(result > t, 1, 0)
                mlb = self.mlbs.get(layer)

                mlb: MultiLabelBinarizer
                tags = mlb.inverse_transform(result_mask)
                values = [
                    result[i, np.where(result_mask[i])][0] for i in range(num_images)
                ]

                out = [
                    [
                        {"tag": t, "value": float(v), "category": layer}
                        for t, v in zip(t_list, v_list)
                    ]
                    for t_list, v_list in zip(tags, values)
                ]

            for i in range(num_images):
                res[i].extend(out[i])

        return res
