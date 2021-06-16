import unittest
from scripts import influx_metrics as imetrics


class TestInfluxMetrics(unittest.TestCase):
    def test_get_config(self):
        expected = {'token', 'org', 'bucket', 'source', 'url'}
        actual = set(imetrics.get_config().keys())

        self.assertEqual(expected, actual)
