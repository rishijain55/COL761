# FP-Tree Data Compression Algorithm

## Overview
This repository contains an implementation of a data compression algorithm using FP trees. The algorithm works by sorting transactions based on the number of items they contain, dividing them into distinct blocks, and utilizing the FP-growth algorithm to extract frequent patterns. These patterns are then compressed in the dataset, creating a more compact representation.

## Algorithm
After finding the optimal threshold support, the transactions are sorted and divided into blocks. In each block, the FP-growth algorithm is applied to mine frequent patterns, which are then targeted for replacement with compressed representations. This process efficiently combines transaction sorting, block division, FP-growth mining, and targeted pattern replacement to achieve a compact yet meaningful compressed dataset.

### Algorithm Steps
1. Sort transactions by the number of items.
2. Divide sorted transactions into blocks.
3. For each block:
   a. Build FP-tree using FP-growth algorithm.
   b. Mine frequent patterns from the FP-tree in order of their length.
   c. For each frequent pattern:
      - If the pattern appears more than once, generate a compressed code and update the mapping.
   d. For each transaction:
      - Replace frequent patterns with compressed codes using the mapping.
      - Append the compressed transaction to the compressed transactions.

## Running Instructions

Clone the repository and navigate to the directory.
Compile the code using the following command:

```bash
$ bash compile.sh
```

### Compression 

To compress a dataset, use the following command:

```bash
$ bash interface.sh C <path/to/dataset.dat> <path/to/output.dat>
```

This command will output a compressed dataset saved in a human-readable format. The compression ratio will also be computed.

### Decompression
To decompress a dataset, use the following command:

```bash
$ bash interface.sh D <path/to/compressed_dataset.dat> <path/to/reconstructed.dat>
```
This command will decompress the dataset, producing the reconstructed dataset. Additionally, any loss penalty will be computed.


