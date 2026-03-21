"""Utilities for counting CSI packets from a specific MAC address.

This module relies on the CSIKit library to read capture files and
counts the frames whose recorded transmitter/source MAC matches a
provided address. It can also be executed as a small CLI utility.
"""
from argparse import ArgumentParser
from typing import Any, Callable, Iterable, Optional

DatasetLoader = Callable[[str], Any]


def _extract_mac(frame: Any) -> Optional[str]:
    """Return the first MAC-like attribute found on a CSIKit frame.

    CSIKit frame objects may expose the transmitter address under different
    attribute names depending on the capture backend. This helper checks a
    handful of common attribute names and returns the first non-empty value.
    """

    for attr in ("source_mac", "transmitter_mac", "addr2", "src_mac", "mac"):
        value = getattr(frame, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _default_dataset_loader() -> DatasetLoader:
    """Resolve a CSIKit dataset loader across library versions.

    Newer CSIKit releases no longer expose ``read_file`` from
    ``CSIKit.reader``. To remain compatible, this helper tries multiple
    import locations and falls back to building a reader via
    ``get_reader`` when necessary.
    """

    try:
        from CSIKit.reader import read_file

        return read_file
    except Exception:
        pass

    get_reader = None
    for candidate in (
        "CSIKit.reader",
        "CSIKit.reader.readers",
    ):
        try:
            module = __import__(candidate, fromlist=["get_reader"])
            get_reader = getattr(module, "get_reader", None)
            if get_reader is not None:
                break
        except Exception:
            continue

    if get_reader is None:
        raise ImportError(
            "CSIKit does not expose read_file or get_reader for loading captures"
        )

    def _load_with_reader(path: str) -> Any:
        reader = get_reader(path)
        if hasattr(reader, "read_file"):
            return reader.read_file(path)
        if hasattr(reader, "read"):
            return reader.read(path)
        raise ImportError("CSIKit reader object did not provide a read method")

    return _load_with_reader


def count_packets_from_mac(
    capture_path: str,
    mac_address: str,
    loader: Optional[DatasetLoader] = None,
) -> int:
    """Count packets for a given MAC address using CSIKit.

    Parameters
    ----------
    capture_path: str
        Path to a CSI capture readable by CSIKit (e.g., ``.dat``, ``.pcap``).
    mac_address: str
        The MAC address to match. Comparison is case-insensitive.
    loader: Optional[Callable[[str], Any]]
        Optional loader used to read the capture. Defaults to a compatible
        CSIKit loader resolved at runtime. Supplying a loader simplifies
        testing and allows custom pipelines.

    Returns
    -------
    int
        Number of frames whose transmitter/source MAC matches ``mac_address``.

    Raises
    ------
    ValueError
        If the dataset does not expose a ``frames`` iterable.
    """

    normalized_mac = mac_address.lower()
    dataset_loader = loader or _default_dataset_loader()

    dataset = dataset_loader(capture_path)
    frames: Optional[Iterable[Any]] = getattr(dataset, "frames", None)
    if frames is None:
        raise ValueError("CSI dataset did not expose a 'frames' iterable")

    count = 0
    for frame in frames:
        source_mac = _extract_mac(frame)
        if source_mac is not None and source_mac.lower() == normalized_mac:
            count += 1
    return count


def count_packets_for_macs(
    capture_path: str, mac_addresses: Iterable[str], loader: Optional[DatasetLoader] = None
) -> tuple[int, dict[str, int]]:
    """Count total frames and occurrences for multiple MAC addresses using CSIKit.

    Parameters
    ----------
    capture_path: str
        Path to a CSI capture readable by CSIKit (e.g., ``.dat``, ``.pcap``).
    mac_addresses: Iterable[str]
        Collection of MAC addresses to count. Comparison is case-insensitive.
    loader: Optional[Callable[[str], Any]]
        Optional loader used to read the capture. Defaults to a compatible
        CSIKit loader resolved at runtime.

    Returns
    -------
    tuple[int, dict[str, int]]
        Total frame count and mapping of normalized MAC addresses to packet counts.
    """

    normalized_macs = [mac.strip().lower() for mac in mac_addresses if mac.strip()]
    dataset_loader = loader or _default_dataset_loader()

    dataset = dataset_loader(capture_path)
    frames: Optional[Iterable[Any]] = getattr(dataset, "frames", None)
    if frames is None:
        raise ValueError("CSI dataset did not expose a 'frames' iterable")

    mac_counts: dict[str, int] = {mac: 0 for mac in normalized_macs}
    total_packets = 0

    for frame in frames:
        total_packets += 1
        if not normalized_macs:
            continue

        source_mac = _extract_mac(frame)
        if source_mac is not None:
            normalized_source = source_mac.lower()
            if normalized_source in mac_counts:
                mac_counts[normalized_source] += 1

    return total_packets, mac_counts


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Count packets from a MAC address in a CSIKit-supported capture.",
    )
    parser.add_argument("capture_path", help="Path to the capture file (e.g., .dat, .pcap)")
    parser.add_argument("mac_address", help="MAC address to count")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    packets = count_packets_from_mac(args.capture_path, args.mac_address)
    print(
        f"Found {packets} packet{'s' if packets != 1 else ''} "
        f"from {args.mac_address} in {args.capture_path}"
    )


if __name__ == "__main__":
    main()
