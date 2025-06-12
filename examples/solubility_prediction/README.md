# Molecular Solubility Prediction

This directory demonstrate our Open Data Scientist's capability on model training and evaluation for a domain-specific task: molecular solubility prediction.

## Data

ESOL.csv is extracted from [envGNN](https://github.com/shang-zhu/envchemGNN) repository by Zhu et al., and the dataset is orginally from Delaney, J. S. (2004).

It includes 1144 chemical compounds along with their SMILES strings, and measured/predicted solubility values (log(solubility:mol/L)). More details can be found in the orignal publication.

## Run the analysis

   ```bash
   # export together api key
   export TOGETHER_API_KEY="your-api-key-here"

   # run the agent
   open-data-scientist --executor tci --write-report
   ```

In the CLI, ESOL.csv needs to be uploaded for data analysis.

## References

Delaney, J. S. (2004). Journal of Chemical Information and Computer Sciences, 44(3), 1000–1005. [[link](https://pubs.acs.org/doi/10.1021/ci034243x)]