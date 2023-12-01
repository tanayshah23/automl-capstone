import random
from torch.utils.data import Dataset
from inputs.dataset import get_feature_set

class MetaTask(Dataset):

    def __init__(self, examples, num_task, k_support, k_query):
        """
        :param examples: list of examples
        :param num_task: number of training tasks.
        :param k_support: number of support sample per task
        :param k_query: number of query sample per task
        """
        self.examples = examples
        random.shuffle(self.examples)

        self.num_task = num_task
        self.k_support = k_support
        self.k_query = k_query
        self.max_seq_length = 128
        self.create_batch(self.num_task)

    def create_batch(self, num_task):
        self.supports = []  # support set
        self.queries = []  # query set

        for b in range(num_task):  # for each task
            # 1.select domain randomly
            domain = random.choice(self.examples)['domain']
            domainExamples = [e for e in self.examples if e['domain'] == domain]

            # 1.select k_support + k_query examples from domain randomly
            selected_examples = random.sample(domainExamples,self.k_support + self.k_query)
            random.shuffle(selected_examples)
            exam_train = selected_examples[:self.k_support]
            exam_test  = selected_examples[self.k_support:]

            self.supports.append(exam_train)
            self.queries.append(exam_test)

    def __getitem__(self, index):
        support_set = get_feature_set(self.supports[index])
        query_set   = get_feature_set(self.queries[index])
        return support_set, query_set

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.num_task