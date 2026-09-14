# PathoMENU


PathoMENU is a framework for accurately predicting the pathogenicity of mutations at metal ion-binding residues. This framework is built on the principle of jointly modeling local coordination geometry with global protein architecture through dual-view geometric equivariant graph neural network, and integrating sequence information and multi-scale structural context through modality contribution-aware fusion mechanism.

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

1. **FoldX**   
   Download from [FoldX](https://foldxsuite.crg.eu/) and place the executable at `tools/foldx`.

2. **Foldseek** 
   Download from [Foldseek](https://github.com/steineggerlab/foldseek) and place the executable at `tools/foldseek`.

3. **ESM1v** 
   Download [`esm1v_t33_650M_UR90S_1.pt`](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt) to `weights/pretrained/esm1v_t33_650M_UR90S_1.pt`.

4. **SaProt**  
   Download [`SaProt_650M_AF2`](https://huggingface.co/westlake-repl/SaProt_650M_AF2) to `weights/pretrained/SaProt_650M_AF2/`.


## Step 4: Run PathoMENU


Using `./example/inputs.csv` as an example, prepare the input data in the following format:

```csv
pdb_id, sequence, pdb_position, sequence_position, metal ion, wt_aa, mut_aa
```
Here, `pdb_id` is the file name containing the protein-metal ion complex structure; `sequence` is the protein amino acid sequence; `pdb_position` is the residue position in the PDB structure; `sequence_position` is the corresponding position in the protein sequence; `metal ion` is the metal ion type; and `wt_aa` and `mut_aa` are the wild-type and mutant amino acids, respectively.


Run the following command to obtain PathoMENU predictions:

```commandline
python scripts/predict.py ./example/inputs.csv
```

The results are saved to `./example/predictions.csv`.
