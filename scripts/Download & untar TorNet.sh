export TORNET_ROOT=/data/tornet 
mkdir -p $TORNET_ROOT && cd $TORNET_ROOT

# 1) Grab 2013–2022 in bulk via Zenodo IDs
#    (for 2013 add Zenodo id its not here)
for id in 12637032 12655151 12655179 12655183 12655187 \
          12655716 12655717 12655718 12655719
do
  zenodo_get $id
  tar -xzf tornet_*.tar.gz && rm tornet_*.tar.gz
done


# 3) Get the catalog
wget -q https://zenodo.org/record/12636522/files/catalog.csv -O catalog.csv

# Quick check:
find $TORNET_ROOT -name '*.nc' | wc -l

# if you have to pull the dataset through some other commands feel free