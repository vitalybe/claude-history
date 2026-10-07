#!/usr/bin/env bash
# Local release build mirroring CI: dynamic ONNX Runtime from Microsoft's
# official release (pyke's CDN used by ort-sys download-binaries 403s).
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

ort_version="$(sed -n 's/^ *ORT_VERSION: *//p' .github/workflows/release.yml)"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) ort_platform=osx-arm64 ;;
  Linux-x86_64) ort_platform=linux-x64 ;;
  *) echo "unsupported platform for prebuilt ONNX Runtime" >&2; exit 1 ;;
esac

ort_dir="target/onnxruntime-$ort_platform-$ort_version"
if [[ ! -d "$ort_dir/lib" ]]; then
  mkdir -p "$ort_dir"
  curl --fail --location --retry 3 \
    "https://github.com/microsoft/onnxruntime/releases/download/v$ort_version/onnxruntime-$ort_platform-$ort_version.tgz" \
    | tar -xz -C "$ort_dir" --strip-components=1
fi

cargo build --release --locked --features release-dynamic-ort "$@"
mkdir -p target/release/lib
find "$ort_dir/lib" -maxdepth 1 \( -type f -o -type l \) -name 'libonnxruntime*' \
  -exec cp -P {} target/release/lib/ \;
echo "built: $PWD/target/release/claude-history"
