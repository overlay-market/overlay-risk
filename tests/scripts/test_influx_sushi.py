import unittest
from scripts import influx_sushi as isushi


class TestInfluxSushi(unittest.TestCase):
    def test_get_config(self):
        expected = {'token', 'org', 'bucket', 'url'}
        actual = set(isushi.get_config().keys())

        self.assertEqual(expected, actual)
