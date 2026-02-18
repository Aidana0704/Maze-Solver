import unittest
from parameterized import parameterized

from load_image import convert_grid_point_to_image_point

class test_load_image(unittest.TestCase):
    @parameterized.expand([((1, 1), (17, 17)), ((2, 3), (33, 49))])
    def test_grid_to_image_space(self, input, expected):
        self.assertEqual(convert_grid_point_to_image_point(input), expected)

if __name__ == '__main__':
    unittest.main()