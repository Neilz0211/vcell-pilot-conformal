#!/usr/bin/env bash
# Run commands on a RunPod pod through the PTY-only proxy. Usage:
#   podsh.sh HOST 'cmd1; cmd2'                 run commands
#   podsh.sh HOST --put LOCAL REMOTE           upload a small text/binary file (base64, chunked lines)
set -e
HOST="$1"; shift
KEY=~/.ssh/runpod_ed25519
if [ "$1" = "--put" ]; then
  LOCAL="$2"; REMOTE="$3"
  {
    echo "rm -f /tmp/up.b64"
    base64 -w 800 "$LOCAL" | while read -r line; do echo "echo '$line' >> /tmp/up.b64"; done
    echo "base64 -d /tmp/up.b64 > $REMOTE && md5sum $REMOTE"
    echo "exit"
  } | timeout 300 ssh -tt -i $KEY -o StrictHostKeyChecking=no -o LogLevel=ERROR "$HOST" 2>&1 | grep -E "^[0-9a-f]{32} " || true
  md5sum "$LOCAL"
else
  printf '%s; exit\n' "$1" | timeout 600 ssh -tt -i $KEY -o StrictHostKeyChecking=no -o LogLevel=ERROR "$HOST" 2>&1 | sed -n '/__BEGIN__/,/__END__/p'
fi
