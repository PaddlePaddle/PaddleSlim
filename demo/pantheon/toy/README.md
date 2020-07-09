## Toy example for Pantheon

See more details about Pantheon in [PaddleSlim/Pantheon](../../../paddleslim/pantheon).

Here implements two teacher models (not trainable, just for demo): teacher1 takes an integer **x** as input and predicts value **2x-1**, see in [run_teacher1.py](run_teacher1.py); teacher2 also takes **x** as input and predicts **2x+1**, see in [run_teacher2.py](run_teacher2.py). They two share a data reader to read a sequence of increasing natural numbers from zero to some positive inter **max_n** as input and generate different knowledge. And the schema keys for  knowledge in teacher1 is [**"x", "2x-1", "result"**], and [**"2x+1", "result"**] for knowledge in teacher2, in which **"result"** is the common schema and the copy of two  predictions respectively. On instantiating the **Student** object, the merging strategy for the common schema **"result"** should be specified, and the schema keys for the merged knowledge will be [**"x", "2x-1", "2x+1", "result"**], with the merged **"result"** equal to **"2x"** when the merging strategy is **"mean"** and **"4x"** when merging strategy is **"sum"**. The student model gets merged knowledge from teachers and prints them out, see in [run_student.py](run_student.py).

The toy "knowledge distillation" system can be launched in three different modes, i.e., offline, online and their hybrid. All three modes should have the same outputs, and the correctness of results can be verified by checking the order and values of outputs.

### Offline

 The two teachers work in offline mode, and start them with given local file paths.

 ```shell
export PYTHONPATH=../../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export NUM_POSTPROCESS_THREADS=10 # default 8
nohup python -u run_teacher1.py --use_cuda true --out_path teacher1_offline.dat > teacher1_offline.log 2>&1&
export CUDA_VISIBLE_DEVICES=2
nohup python -u run_teacher2.py --use_cuda true --out_path teacher2_offline.dat > teacher2_offline.log 2>&1&
 ```
 After the two executions both finished, start the student model with the two generated knowledge files.

 ```shell
export PYTHONPATH=../../../:$PYTHONPATH
 python -u run_student.py \
        --in_path0 teacher1_offline.dat \
        --in_path1 teacher2_offline.dat
 ```


### Online

The two teachers work in online mode, and start them with given TCP/IP ports. Please make sure that the ICP/IP ports are available.

```shell
export PYTHONPATH=../../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
nohup python -u run_teacher1.py --use_cuda true --out_port 8080  > teacher1_online.log 2>&1&
export CUDA_VISIBLE_DEVICES=1,2
nohup python -u run_teacher2.py --use_cuda true --out_port 8081  > teacher2_online.log 2>&1&
```
Start the student model with the IP addresses that can reach the ports of the two teacher models, e.g., in the same node

```shell
export PYTHONPATH=../../../:$PYTHONPATH
python -u run_student.py \
         --in_address0 127.0.0.1:8080 \
         --in_address1 127.0.0.1:8081 \
```
**Note:** in online mode, the starting order of teachers and the sudent doesn't matter, and they will wait for each other to establish connection.

### Hybrid of offline and online

One teacher works in offline mode and another one works in online mode. This time, start the offline teacher first. After the offline knowledge file gets well prepared, start the online teacher and the student at the same time.
