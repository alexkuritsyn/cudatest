rm *.jpg*
#scp -r kodiak.baylor.edu:/home/kuritcyna/code/mpplabs/testing/frames/ .
if [ -d frames ]
then
echo "frames directory exists"
else
scp -r kodiak.baylor.edu:/home/kuritcyna/code/mpplabs/testing/frames/ .
fi
echo "start script"
~/code/mpplabs/testing/device-query
echo "copy back"
scp *.jpg* kodiak.baylor.edu:/home/kuritcyna/code/mpplabs/testing/diff/
