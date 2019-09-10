# Example use of this repository.

Assuming the data is structured that way:

~/data/us/forms => Complete original forms.

All commands are assumed to be runned from the root of the repository.
The repository is assumed a first level folder in the home of the user.

## Data Process

1. Aligned Urgence Santé Scanned Forms.

## 0. Configuration

pip3 install img-align

## 1. Alignment

```{bash}
# Create alignment directory if missing
mkdir -p ../data/us/aligned-forms
# Align Forms
python3 img-align/ --ref "../data/us/forms/21543780.png" --img "../data/us/forms/*.png" --out "../data/us/aligned-forms"
```

