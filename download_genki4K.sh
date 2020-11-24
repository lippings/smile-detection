#!/bin/bash
# Download GENKI 4K dataset and extract to ./genki4k

echo ""
echo "Downloading..."
echo ""

wget https://inc.ucsd.edu/mplab/databases/GENKI-4K.zip

echo ""
echo "Unzipping..."
echo ""

unzip GENKI-4K.zip -d genki4k
