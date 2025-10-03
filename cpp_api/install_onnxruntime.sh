set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNXRUNTIME_DIR="${SCRIPT_DIR}/onnxruntime"
VERSION="1.16.1"
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-linux-x64-gpu-${VERSION}.tgz"
ARCHIVE_NAME="onnxruntime-linux-x64-gpu-${VERSION}.tgz"
EXTRACTED_DIR="onnxruntime-linux-x64-gpu-${VERSION}"

echo "Downloading ONNX Runtime v${VERSION}..."
if [ ! -f "$ARCHIVE_NAME" ]; then
    wget "$DOWNLOAD_URL"
else
    echo "Archive already exists, skipping download."
fi

echo "Extracting ONNX Runtime..."
tar -xzf "$ARCHIVE_NAME"

echo "Renaming directory to 'onnxruntime'..."
if [ -d "$ONNXRUNTIME_DIR" ]; then
    echo "Removing existing onnxruntime directory..."
    rm -rf "$ONNXRUNTIME_DIR"
fi
mv "$EXTRACTED_DIR" "$ONNXRUNTIME_DIR"

echo "Installing ONNX Runtime to system..."

if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    echo "Error: onnxruntime directory not found at $ONNXRUNTIME_DIR"
    exit 1
fi

echo "Copying headers to /usr/local/include/onnxruntime..."
sudo mkdir -p /usr/local/include/onnxruntime
sudo cp -r "${ONNXRUNTIME_DIR}/include/"* /usr/local/include/onnxruntime/

echo "Copying libraries to /usr/local/lib..."
sudo cp "${ONNXRUNTIME_DIR}/lib/"* /usr/local/lib/

echo "Updating library cache..."
sudo ldconfig

echo ""
echo "Verifying installation..."
if ldconfig -p | grep onnxruntime > /dev/null; then
    echo "✓ ONNX Runtime successfully installed!"
    echo ""
    echo "Library locations:"
    ldconfig -p | grep onnxruntime
else
    echo "✗ Installation verification failed"
    exit 1
fi

echo ""
echo "Installation complete!"
