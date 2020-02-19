# 多进程蒸馏

## Teacher

pantheon.Teacher() [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/teacher.py#L78)

: The class defined for the teacher model. Generate knowledge data and transfer them to the student model.

**Args:**

- **out\_path (str|None)** - The path to dump knowledge data for offline mode.

- **out\_port (int|None)** - The IP port number to send out knowledge for online mode, should be unique when launching multiple teachers in the same node.

**Return:** An object of class Teacher


pantheon.Teacher.start() [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/teacher.py#L133)

: Start teacher service, sychronize with student and launch the thread
  to monitor commands from student.

**Args:** None

**Return:** None


pantheon.Teacher.send(data) [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/teacher.py#L181)

: Send one data object to student.

**Args:**

- **data (Python data):** - The data to be sent, can be any type of Python data object.

**Return:** None


pantheon.Teacher.recv() [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/teacher.py#L196)

: Recieve one data object from student.

**Args:** None

**Return:**

- The received data, can be any type of Python data object.


pantheon.Teacher.dump(knowledge) [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/teacher.py#L214)

: Dump one batch knowledge data into the output file, only used in the offline mode.

**Args:**

- **knowledge (dict):** - The knowledge data to be dumped.  

**Return:** None


pantheon.Teacher.start\_knowledge\_service(feed\_list, schema, program, reader\_config, exe, buf\_size=10, times=1) [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/teacher.py#L259)

: Start the knowledge service to generate and transfer knowledge data. In GPU mode, the devices to execute knowledge prediction will be determined by the
  environment variable **FLAGS\_selected\_gpus**, or by **CUDA\_VISIBLE\_DEVICES** if it is not set, and by **CPU\_NUM** (default 1) in CPU mode. Only supported in static graph.

 **Args:**

 - **feed\_list (list):** - A list of feed Variables or their names for the
                              input teacher Program.
 - **schema (dict):** - A dict to specify keys and fetched Variables  
                        to generate knowledge.
 - **program (fluid.Program):** - Inference Program of the teacher model.
 - **reader\_config (dict):** - The config for data reader. Support all the three types of generators used by [fluid.io.PyReader](https://www.paddlepaddle.org.cn/documentation/docs/en/api/io/PyReader.html) and [fluid.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/en/api/io/DataLoader.html#dataloader), and their configs contain the key-value pair of the generator type and a generator object, plus other necessary argument pairs. See the following:

     1) **sample generator:**

     ```
     reader_config={"sample_generator": some_sample_generator,
                    "batch_size": batch_size, "drop_last": drop_last}
     # drop_last set to True by default
     ```

     2) **sample list generator:**

     ```
     reader_config={"sample_list_generator": some_sample_list_generator}
     ```

     3) **batch generator:**

     ```
     reader_config={"batch_generator": some_batch_genrator}
     ```

     The trial to parse config will be in the order of 1) -> 3), and any other unrelated keys in these configs will be ignored.

- **exe (fluid.Executor):** The executor to run the input program.
- **buf\_size (int):** The size of buffers for data reader and knowledge
                            writer on each device.
- **times (int):** The maximum repeated serving times, default 1. Whenever
                         the public method **get\_knowledge\_generator()** in **Student**
                         object called once, the serving times will be added one,
                         until reaching the maximum and ending the service. Only
                         valid in online mode, and will be ignored in offline mode.

**Return:** None

**Examples:**

```python
import paddle
import paddle.fluid as fluid
from paddleslim.pantheon import Teacher

startup = fluid.Program()
program = fluid.Program()
with fluid.program_guard(program, startup):
    images = fluid.data(
            name='pixel', shape=[None, 3 * 32 * 32], dtype='float32')
    labels = fluid.data(name='label', shape=[None, 1], dtype='int64')
    logits = fluid.layers.fc(input=images, size=10)
    loss = fluid.layers.softmax_with_cross_entropy(logits, labels)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup)

train_reader = paddle.batch(
        paddle.dataset.cifar.train10(), batch_size=32)

teacher = Teacher(out_path="example_knowledge.dat", # offline mode
                  #out_port=5000                    # online mode
                  )
teacher.start()

teacher.start_knowledge_service(
    feed_list=[images, labels],
    schema={"logits": logits,
            "labels": labels},
    program=program,
    reader_config={"sample_list_generator": train_reader},
    exe=exe)
```

!!! note "Note"
    This example should be run with the example of class **Student**.


## Student

pantheon.Student(merge_strategy=None) [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/student.py#L34)

: The class defined for the student model. Receive knowledge data from
    teacher model and carry out knowledge merging.  

 **Args:**

 - **merge\_strategy (dict|None):** - A dict whose keys are the common schemas shared by different teachers, and each corresponding value specifies the merging strategy for different schemas respectively, supporting **sum** and **mean** now.

**Return:** An object of class Student.


pantheon.Student.register\_teacher(in\_path=None, in\_address=None) [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/student.py#L72)

: Register one teacher model and assign the order number to it as  its id, with the file path (offline mode) or IP address (online  mode) that the teacher model writes knowledge data to.

**Args:**

- **in\_path (str|None):** The input file path. Default None.
- **in\_address (str|None):** The input IP address, in the format "&lt;IP\_address&gt;:&lt;IP\_port&gt;" (e.g. "127.0.0.1:8080"). Default None.

**Return:**  None


pantheon.Student.start() [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/student.py#L213)

: End teachers' registration and synchronize with all of them.

**Args:** None

**Return:**  None

pantheon.Student.send(self, data, teacher_ids=None) [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/student.py#L240)

: Send data to teachers.

**Args:**

- **data (Python data):** - A Python data object to be sent.
- **teacher_ids (list|None):** - A list of teacher ids to send data. If set to None, send the data to all teachers. Default None.

**Return:**  None

pantheon.Student.recv(teacher_id) [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/student.py#L262)

: Receive data from one teacher.

 **Args:**

- **teacher\_id (int):** - The id of teacher that receives data from.

**Return:**  

- The received data object.

pantheon.Student.get\_knowledge\_desc() [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/student.py#L283)

 : Get description for knowledge, including shape, data type and lod level for each schema.

 **Args:** None

 **Return:**  

 - Knowledge description, which is a dict.


pantheon.Student.get\_knowledge\_qsize() [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/student.py#L318)

 : Get the real-time size of knowledge queue. If this size is denoted as
   **qsize**, it means that there are **qsize** batch knowledge data
   already pushed into knowledge queue and waiting for the knowledge
   generator to pop out. It's dynamic and limited up to 100, the capacity
   of the knowledge queue.

 **Args:** None

 **Return:**  

 - The real-time size of knowledge queue.

pantheon.Student.get\_knowledge\_generator(batch\_size, drop\_last=False) [source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/pantheon/student.py#L334)

: Get the generator for knowledge data, return None if last generator doesn't finish yet.

**Args:**

- **batch\_size (int):** - The batch size of returned knowledge data.
- **drop\_last (bool):** - Whether to drop the last batch if its size is less than batch size.

**Return:**

- The wrapper of knowledge data generator.

**Examples:**

```python
from paddleslim.pantheon import Student

student = Student()

student.register_teacher(in_path="example_knowledge.dat",  # offline mode
                         #in_address="127.0.0.1:5000"      # online mode
                         )
student.start()

knowledge_desc = student.get_knowledge_desc()
data_generator = student.get_knowledge_generator(
    batch_size=128, drop_last=True)

# get knowledge data
for knowledge in data_generator():
    print("knowledge queue size: {}".format(student.get_knowledge_qsize()))

    # do something else
```

!!! note "Note"
    This example should be run with the example of class **Teacher**.
