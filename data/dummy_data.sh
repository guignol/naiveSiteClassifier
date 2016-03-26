key=82fd23d0
nb_iterations=100
#hdfs dfs -mkdir -p /user/cloudera/url
mkdir url
rm -f url/url_*
for i in `seq 1 $nb_iterations`
do
	file_name="url/url_$i"
	curl "https://www.mockaroo.com/7f658520/download?count=5000&key=$key" > $file_name
#	hdfs dfs -put $file_name /user/cloudera/url
#	rm -f $file_name
done
