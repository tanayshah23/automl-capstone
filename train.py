import os
import time
import torch
import random
import numpy as np
from dataset import MetaTask

def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]

def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def train(args, learner, train_examples, test):
    global_step = 0
    for epoch in range(args.maml.meta_epoch):
        train = MetaTask(train_examples, num_task=args.maml.num_train_task, k_support=args.maml.k_support, k_query=args.maml.k_query)
        db = create_batch_of_tasks(train, is_shuffle = True, batch_size = args.maml.outer_batch_size)
        for step, task_batch in enumerate(db):
            f = open(os.path.join(args.output_dir, args.maml.log_file), 'a')
            acc = learner(task_batch)
            print('Step:', step, '\ttraining Acc:', acc)
            f.write(str(acc) + '\n')

            if global_step % 20 == 0:
                random_seed(123)
                print("\n-----------------Testing Mode-----------------\n")
                db_test = create_batch_of_tasks(test, is_shuffle = False, batch_size = 1)
                acc_all_test = []

                for test_batch in db_test:
                    acc = learner(test_batch, training = False)
                    acc_all_test.append(acc)

                print('Step:', step, 'Test F1:', np.mean(acc_all_test))
                f.write('Test' + str(np.mean(acc_all_test)) + '\n')

                random_seed(int(time.time() % 10))

            global_step += 1
            f.close()