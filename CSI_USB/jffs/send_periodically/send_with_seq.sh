#!/bin/sh
set -eu

RAWPERF="/jffs/send_periodically/rawperf"
PKTFILE="/jffs/send_periodically/packetnode1x1BP.dat"

IFACE="${1:-}"
COUNT="${2:-}"
DELAY_US="${3:-2000}"      # keep your original 2000 us default
SEQ_START="${4:-0}"        # starting 12-bit seq number
SEQ_OFFSET="${5:-30}"      # byte offset of Sequence Control in *this* .dat

if [ -z "$IFACE" ] || [ -z "$COUNT" ]; then
  echo "Usage: $0 <interface> <count> [delay_us=2000] [seq_start=0] [seq_offset=30]"
  echo "Example: $0 eth6 1000 2000 0 30"
  exit 1
fi

# Try to use usleep if present; fallback to sleep (may be coarse on busybox)
do_usleep() {
  us="$1"
  if command -v usleep >/dev/null 2>&1; then
    usleep "$us"
  elif command -v busybox >/dev/null 2>&1 && busybox --list 2>/dev/null | grep -q '^usleep$'; then
    busybox usleep "$us"
  else
    # Fallback: sleep in seconds (may not support fractions on some systems)
    # 2000 us = 0.002 s
    sec="$(awk "BEGIN { printf(\"%.6f\", $us/1000000) }")"
    sleep "$sec" 2>/dev/null || true
  fi
}

# Write 2 bytes (little-endian) at SEQ_OFFSET
write_seq() {
  seq12="$1"  # 0..4095
  seq_ctrl=$(( (seq12 & 4095) << 4 ))     # fragment=0
  lo=$(( seq_ctrl & 255 ))
  hi=$(( (seq_ctrl >> 8) & 255 ))

  # Build two raw bytes
  b1=$(printf '\\x%02x' "$lo")
  b2=$(printf '\\x%02x' "$hi")

  # Write into file at offset
  # shellcheck disable=SC2059
  printf "$b1$b2" | dd of="$PKTFILE" bs=1 seek="$SEQ_OFFSET" conv=notrunc 2>/dev/null
}

# Basic sanity checks
if [ ! -x "$RAWPERF" ]; then
  echo "Error: rawperf not executable at: $RAWPERF"
  exit 1
fi
if [ ! -f "$PKTFILE" ]; then
  echo "Error: packet file not found at: $PKTFILE"
  exit 1
fi

# Ensure file is long enough
FILESIZE=$(wc -c < "$PKTFILE" | tr -d ' ')
if [ "$FILESIZE" -lt $((SEQ_OFFSET + 2)) ]; then
  echo "Error: $PKTFILE too small ($FILESIZE bytes) for seq_offset=$SEQ_OFFSET"
  exit 1
fi

i=0
while [ "$i" -lt "$COUNT" ]; do
  seq=$(( (SEQ_START + i) & 4095 ))
  write_seq "$seq"

  # Send exactly ONE frame using the same file.
  # Keeping -t and -q for compatibility; -n 1 ensures a single packet.
  "$RAWPERF" -i "$IFACE" -n 1 -f "$PKTFILE" -t "$DELAY_US" -q 1 >/dev/null 2>&1

  # Optional pacing between separate rawperf invocations (not the intra-rawperf -t)
  do_usleep "$DELAY_US"

  i=$((i + 1))
done

echo "Done: sent $COUNT packets, seq_start=$SEQ_START, seq_offset=$SEQ_OFFSET"
