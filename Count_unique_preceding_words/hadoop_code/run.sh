rm -r out
rm -r hdfs
mkdir hdfs
hdfs dfs -put Moby-Dick.txt hdfs/Moby-Dick.txt
hadoop jar /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
    -files mapper.py \
    -mapper "python3 mapper.py" \
    -file reducer.py \
    -reducer "python3 reducer.py" \
    -input hdfs/Moby-Dick.txt \
    -output out
cat out/part-00000

