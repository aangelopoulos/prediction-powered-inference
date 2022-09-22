cd ~/mai_datasets/plankton
mkdir merged
#wget https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7342/2006.zip?sequence=1&isAllowed=y
#unzip 2006.zip?sequence=1 | pv -l >/dev/null
#wget https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7343/2007.zip?sequence=1&isAllowed=y
#unzip 2007.zip?sequence=1 | pv -l >/dev/null
#wget https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7345/2008.zip?sequence=1&isAllowed=y
#unzip 2008.zip?sequence=1 | pv -l >/dev/null
#wget https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7346/2009.zip?sequence=1&isAllowed=y
#unzip 2009.zip?sequence=1 | pv -l >/dev/null
#wget https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7348/2010.zip?sequence=1&isAllowed=y
#unzip 2010.zip?sequence=1 | pv -l >/dev/null
#wget https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7347/2011.zip?sequence=1&isAllowed=y
#unzip 2011.zip?sequence=1 | pv -l >/dev/null
#wget https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7344/2012.zip?sequence=1&isAllowed=y
#unzip 2012.zip?sequence=1 | pv -l >/dev/null
#wget https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7349/2013.zip?sequence=1&isAllowed=y
#unzip 2013.zip?sequence=1 | pv -l >/dev/null
#wget https://darchive.mblwhoilibrary.org/bitstream/handle/1912/7350/2014.zip?sequence=1&isAllowed=y
#unzip 2014.zip?sequence=1 | pv -l >/dev/null

rsync -a --info=progress2 --link-dest=../2006 ./2006/ ./merged/
rsync -a --info=progress2 --link-dest=../2007 ./2007/ ./merged/
rsync -a --info=progress2 --link-dest=../2008 ./2008/ ./merged/
rsync -a --info=progress2 --link-dest=../2009 ./2009/ ./merged/
rsync -a --info=progress2 --link-dest=../2010 ./2010/ ./merged/
rsync -a --info=progress2 --link-dest=../2011 ./2011/ ./merged/
rsync -a --info=progress2 --link-dest=../2012 ./2012/ ./merged/
rsync -a --info=progress2 --link-dest=../2013 ./2013/ ./merged/
rsync -a --info=progress2 --link-dest=../2014 ./2014/ ./merged/
cd -
