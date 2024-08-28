from abc import *
from loguru import logger
from probables import BloomFilter, CountMinSketch
def get_counter(args):
    if args['counter_type'] == 'array':
        class_num = 10
        if args['dataset'] =='cifar100':
            class_num = 100
        return SimpleCounter(class_num,lambda x : x,args)
    
    elif args['counter_type'] == 'cmsketch':
        return CMSketch(args)
    elif args['counter_type'] =='bloomfilter':
        return Bloomfilter(args)
class Counter:

    @abstractmethod
    def get_counter(self):
        raise NotImplemented
    
    @abstractmethod
    def add_element(self,elment):
        raise NotImplemented
    
    @abstractmethod
    def add_elements(self,elements):
        raise NotImplemented

class SimpleCounter:

    def __init__(self,capacity,hash_func,args) -> None:
        self.capacity = capacity
        self.data = [0]*self.capacity
        self.hash_func = hash_func
        self.args = args
    def get_counter(self):
        return self.data 
    
    def add_element(self,element):
        pos = (self.hash_func(element))%len(self.data)
        self.data[pos] += 1

    def add_elements(self,elements):
        for el in elements:
            self.add_element(el)
    def merge(self,counter):
        for i,val  in enumerate(counter.data):
            self.data[i] += val
    def delete(self,element):
        pos = (self.hash_func(element))%len(self.data)
        self.data[pos] -= 1
        assert(self.data[pos]>=0)

class CMSketch(SimpleCounter):
    def __init__(self,args) -> None:
        super().__init__(0,None,args)
        width=6
        depth=2
        if self.args['dataset'] == 'cifar100':
            width =20
            depth =3
        self.sketch = CountMinSketch(width=width,depth=depth)
        self.delete_sketch = CountMinSketch(width=width,depth=depth)

    
    def add_element(self, element):
        self.sketch.add(str(element))

    def delete(self, element):
        self.delete_sketch.add(str(element))
    
    def get_counter(self):
        class_num = 10
        if self.args['dataset']=='cifar100':
            class_num = 100
        counter = [0]*class_num
        for i in range(class_num):
            counter[i] = (self.sketch.check(str(i))-self.delete_sketch.check(str(i))) 
        return counter
    def merge(self,counter):
        self.sketch.join(counter.sketch)
        self.delete_sketch.join(counter.delete_sketch)
    
class Bloomfilter(SimpleCounter):
    def __init__(self,args) -> None:
        super().__init__(0,None,args)

        est_elments = 10
        if self.args['dataset'] =='cifar100':
            est_elments = 100
        self.blm = BloomFilter(est_elements=est_elments,false_positive_rate=0.05)

    def add_element(self, element):
        self.blm.add(str(element))

    def delete(self, element):
        logger.info("bloom delete")
    
    def get_counter(self):
        class_num = 10
        if self.args['dataset'] == 'cifar100':
            class_num  =100
        counter = [0]*class_num
        for i in range(class_num):
            if self.blm.check(str(i)):
                counter[i] = 1
        return counter
    
    def merge(self,counter):
        self.blm.union(counter.blm)
    