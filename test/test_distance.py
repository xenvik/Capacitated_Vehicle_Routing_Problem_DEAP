import unittest
import math
from text_to_json import calculate_distance
from deap_model import input_instance


class TestDistance(unittest.TestCase):

    def test_distance(self):
        loaded_instance = input_instance("./data/json/Input_Data.json")
        client7, client8 = loaded_instance["client_7"], loaded_instance["client_8"]
        calculated_result = calculate_distance(client7, client8)
        math_result = math.sqrt((40-38)**2 + (66-68)**2)
        self.assertEqual(calculated_result, math_result)

if __name__ == '__main__':
    unittest.main()