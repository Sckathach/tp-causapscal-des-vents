# TP _Causapscal des Vents_

## Installation
> Python 3.12 (ou 3.13 à vos risques et périls (devrait marcher tho))

**Préparez l'environment:**

<details open>
<summary>Avec miniconda (recommandé)</summary>
Installation de Miniconda: 

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

Sourcez ou relancez votre shell:
```shell
source ~/.bashrc
```

Créez l'environment avec Python 3.12:
```shell
conda create -n tp-du-tres-grand-sckatache python=3.12 -y
conda activate tp-du-tres-grand-sckatache
```
</details>

<details>
<summary>Avec python venv</summary>

```shell 
python -m venv .venv 
source .venv/bin/activate
```
</details>


**Puis installez les dépendences:**

```shell
pip install poetry 
poetry install --with notebook
```



