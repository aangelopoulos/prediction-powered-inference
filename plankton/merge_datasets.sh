cd data/
mkdir merged
rsync -a --info=progress2 ./2006/ ./merged/
rsync -a --info=progress2 ./2007/ ./merged/
rsync -a --info=progress2 ./2008/ ./merged/
rsync -a --info=progress2 ./2009/ ./merged/
rsync -a --info=progress2 ./2010/ ./merged/
rsync -a --info=progress2 ./2011/ ./merged/
rsync -a --info=progress2 ./2012/ ./merged/
rsync -a --info=progress2 ./2013/ ./merged/
rsync -a --info=progress2 ./2014/ ./merged/
cd ..
