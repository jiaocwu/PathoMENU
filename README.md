# PathoMENU

***

## Predicting the pathogenicity of metal-binding site variants with sequence and structure features

***

PathoMENU is a deep learning framework for predicting the pathogenicity of missense variants associated with protein metal binding. It uses a dual-view equivariant geometric encoder for precise structural modeling and a modality contribution-aware fusion strategy to adaptively integrate structural and sequence representations

## Step 1: Get PathoMENU

Clone this repository and enter the PathoMENU directory:

```commandline
git clone https://github.com/jiaocwu/PathoMENU.git
cd PathoMENU
```

## Step 2: Build required dependencies

We recommend using [Anaconda](https://www.anaconda.com/download) to install the required dependencies. Run the installation script to create and activate an environment named `PathoMENU`:

```commandline
source install.sh
```



## Step 3: Download third-party packages

The following third-party packages are required to generate PathoMENU features:

1. **FoldX** — generates mutant protein structures.  
   Download from [FoldX Suite](https://foldxsuite.crg.eu/) and place the executable at `tools/foldx`.

2. **Foldseek** — generates structure-aware sequences for SaProt.  
   Download from [Foldseek](https://github.com/steineggerlab/foldseek) and place the executable at `tools/foldseek`.

3. **ESM1v** — generates protein sequence embeddings.  
   Download [`esm1v_t33_650M_UR90S_1.pt`](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt) to `weights/pretrained/esm1v_t33_650M_UR90S_1.pt`.

4. **SaProt** — generates protein structure embeddings.  
   Download [`SaProt_650M_AF2`](https://huggingface.co/westlake-repl/SaProt_650M_AF2) to `weights/pretrained/SaProt_650M_AF2/`.


## Step 4: Run PathoMENU


Using `./example/inputs.csv` as an example, prepare the input data in the following format:

```csv
pdb_id, sequence, pdb_position, sequence_position, metal ion, wt_aa, mut_aa
```
Here, `pdb_id` is the file name containing the protein-metal ion complex structure; `sequence` is the protein amino acid sequence; `pdb_position` is the residue position in the PDB structure; `sequence_position` is the corresponding position in the protein sequence; `metal ion` is the metal ion type; and `wt_aa` and `mut_aa` are the wild-type and mutant amino acids, respectively.


Run the following command to obtain PathoMENU predictions:

```commandline
python scripts/predict_variants.py ./example/inputs.csv
```

The results are saved to `./example/predictions.csv`.
