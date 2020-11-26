#!/bin/bash
# Download GENKI 4K dataset and extract to ./genki4k

echo ""
echo "Downloading..."
echo ""

wget https://inc.ucsd.edu/mplab/databases/GENKI-R2009a.zip

echo ""
echo "Unzipping..."
echo ""

unzip GENKI-R2009a.zip -d genki4k
