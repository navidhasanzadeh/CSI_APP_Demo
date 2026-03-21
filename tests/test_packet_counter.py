import unittest
from typing import Any, Iterable

from packet_counter import count_packets_for_macs, count_packets_from_mac


class DummyFrame:
    def __init__(self, **attributes: str) -> None:
        for key, value in attributes.items():
            setattr(self, key, value)


class DummyDataset:
    def __init__(self, frames: Iterable[Any]) -> None:
        self.frames = list(frames)


class PacketCounterTests(unittest.TestCase):
    def test_counts_packets_with_source_mac(self) -> None:
        frames = [
            DummyFrame(source_mac="aa:bb:cc:dd:ee:ff"),
            DummyFrame(source_mac="11:22:33:44:55:66"),
            DummyFrame(source_mac="AA:BB:CC:DD:EE:FF"),
        ]
        loader = lambda path: DummyDataset(frames)

        result = count_packets_from_mac("/tmp/fake.dat", "aa:bb:cc:dd:ee:ff", loader)

        self.assertEqual(result, 2)

    def test_counts_packets_with_transmitter_mac_attribute(self) -> None:
        frames = [
            DummyFrame(transmitter_mac="aa:bb:cc:dd:ee:ff"),
            DummyFrame(transmitter_mac="00:00:00:00:00:00"),
        ]
        loader = lambda path: DummyDataset(frames)

        result = count_packets_from_mac("/tmp/fake.dat", "aa:bb:cc:dd:ee:ff", loader)

        self.assertEqual(result, 1)

    def test_ignores_frames_without_matching_mac(self) -> None:
        frames = [
            DummyFrame(source_mac="11:22:33:44:55:66"),
            DummyFrame(addr2="77:88:99:AA:BB:CC"),
        ]
        loader = lambda path: DummyDataset(frames)

        result = count_packets_from_mac("/tmp/fake.dat", "aa:bb:cc:dd:ee:ff", loader)

        self.assertEqual(result, 0)

    def test_raises_when_frames_missing(self) -> None:
        class IncompleteDataset:
            pass

        loader = lambda path: IncompleteDataset()

        with self.assertRaises(ValueError):
            count_packets_from_mac("/tmp/fake.dat", "aa:bb:cc:dd:ee:ff", loader)

    def test_counts_packets_for_multiple_macs(self) -> None:
        frames = [
            DummyFrame(source_mac="aa:bb:cc:dd:ee:ff"),
            DummyFrame(source_mac="11:22:33:44:55:66"),
            DummyFrame(source_mac="aa:bb:cc:dd:ee:ff"),
        ]
        loader = lambda path: DummyDataset(frames)

        total, counts = count_packets_for_macs(
            "/tmp/fake.dat", ["aa:bb:cc:dd:ee:ff", "11:22:33:44:55:66"], loader
        )

        self.assertEqual(total, 3)
        self.assertEqual(counts["aa:bb:cc:dd:ee:ff"], 2)
        self.assertEqual(counts["11:22:33:44:55:66"], 1)

    def test_counts_zero_when_no_macs_provided(self) -> None:
        frames = [DummyFrame(source_mac="aa:bb:cc:dd:ee:ff")]
        loader = lambda path: DummyDataset(frames)

        total, counts = count_packets_for_macs("/tmp/fake.dat", [], loader)

        self.assertEqual(total, 1)
        self.assertEqual(counts, {})


if __name__ == "__main__":
    unittest.main()
