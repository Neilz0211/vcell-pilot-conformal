#!/usr/bin/env bash
# Pod-side setup for the GEARS 12-pair run: python deps (torch comes with the image) + public data.
mkdir -p /root/vc/src /root/vc/data /root/vc/results /root/vc/gears
cd /root/vc/data
( pip install -q torch_geometric cell-gears==0.1.2 scanpy anndata scikit-learn dcor > /root/pip_gears.log 2>&1; echo PIPDONE >> /root/setup.log ) &
wget -q https://ndownloader.figshare.com/files/35780870 -O K562_essential_normalized_bulk_01.h5ad
wget -q https://ndownloader.figshare.com/files/35775512 -O rpe1_normalized_bulk_01.h5ad
wget -q http://current.geneontology.org/annotations/goa_human.gaf.gz -O goa_human.gaf.gz
echo DATADONE >> /root/setup.log
wait
python -c "import torch, torch_geometric, gears; print('OK', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" >> /root/setup.log 2>&1
echo SETUPDONE >> /root/setup.log
