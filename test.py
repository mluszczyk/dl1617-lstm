from unittest import TestCase

from train import pad, crop_image


class TestFuncs(TestCase):
    def test_pad(self):
        array = [1, 2, 3, 4, 5]
        item = pad(array, 7)
        self.assertEqual(
            list(item),
            [1, 2, 3, 4, 5, 0, 0]
        )

    def test_crop(self):
        image = [[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]]
        cropped = crop_image(image, 2, 2)

        self.assertEqual(
            cropped.tolist(),
            [[2, 3], [6, 7]]
        )
