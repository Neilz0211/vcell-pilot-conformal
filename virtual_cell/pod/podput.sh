#!/usr/bin/env bash
# Upload a file to a RunPod pod through the PTY-only SSH proxy, in base64 heredoc chunks (faster than podsh.sh --put).
# usage: podput.sh HOST LOCAL REMOTE [CHUNK_MB=4] [PARALLEL=6]
set -e
HOST="$1"; LOCAL="$2"; REMOTE="$3"; CH=$(( ${4:-4} * 1024 * 1024 ))
KEY=~/.ssh/runpod_ed25519
TMP=$(mktemp -d); split -b $CH -d -a 4 "$LOCAL" "$TMP/part."
ssh_run() { timeout 900 ssh -tt -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$HOST" > /dev/null 2>&1; }
printf 'rm -rf /tmp/upl; mkdir -p /tmp/upl; exit\n' | ssh_run
PAR=${5:-6}
for p in "$TMP"/part.*; do
  n=$(basename "$p")
  ( { echo "stty -echo; cat > /tmp/upl/$n.b64 <<'XEOF'"; base64 -w 1000 "$p"; echo "XEOF"; echo "base64 -d /tmp/upl/$n.b64 > /tmp/upl/$n && rm /tmp/upl/$n.b64; exit"; } | ssh_run; echo -n "." ) &
  while [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 1; done
done
wait; echo
printf 'cat /tmp/upl/part.* > %s && rm -rf /tmp/upl; echo __BEGIN__; md5sum %s; echo __END__; exit\n' "$REMOTE" "$REMOTE" \
  | timeout 900 ssh -tt -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$HOST" 2>&1 | tr -d '\r' | grep -E "^[0-9a-f]{32} " || true
md5sum "$LOCAL"; rm -rf "$TMP"
