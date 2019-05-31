import sys
sys.path.append("..")
import numpy as np
from loguru import logger
from lsh.hash_family import E2Family
import random
import networkx as nx
from lsh.lsh import LSH

class node(object):
    def __init__(self,obj, father, lchild=None, rchild=None, isleaf=False):
        self.father=father
        self.obj=obj
        self.lchild=lchild
        self.rchild=rchild
        self.isleaf=isleaf

    def up(self):
        return self.father

    def query(self):
        return self.obj

    def ldown(self, create=False):
        if self.isleaf:
            logger.error("Leaf node do not have left child")
        new=False
        if create: 
            if self.lchild is None:
                self.lchild=node('0', self)
                new=True
            elif self.lchild.isleaf:
                obj=self.lchild.obj
                p=obj[0][obj[-1]]
                obj[-1]+=1
                if p=='0':
                    newnode=node('0', self, self.lchild)
                    self.lchild.father=newnode
                    self.lchild=newnode
                else:
                    newnode=node('1', self, self.lchild)
                    self.lchild.father=newnode
                    if not (self.rchild is None):
                        logger.error("conflict")
                    self.rchild=newnode
                    self.lchild=node('0', self)
                    new=True
                
        return self.lchild, new

    def rdown(self, create=False):
        if self.isleaf:
            logger.error('Leaf node do not have right child')
        new=False
        if create:
            if self.rchild is None:
                self.rchild=node('1', self)
                new=True
        return self.rchild, new

    def add_lchild(self, Node):
        if self.isleaf:
            logger.error("Leaf node cannot add child")
        if not( self.lchild is None):
            logger.error('lchild already exist')
        self.lchild=Node

    def add_rchild(self, Node):
        if self.isleaf:
            logger.error('Leaf node cannnot add child')
        self.rchild=Node

    def insert(self, hashcode, term, depth):
        if self.lchild is None:
            if isinstance(term, list):
                term=term
            else:
                term=[term]
            newnode=node([hashcode, term, depth], self, isleaf=True)
            self.lchild=newnode
            return
        if depth>=len(hashcode):
            if not self.lchild.isleaf:
                logger.error("Lchild is not leaf")
            if isinstance(term, list):
                self.lchild.obj[1].extend(term)
            else:
                self.lchild.obj[1].append(term)
            return 
        q=hashcode[depth]
        if q=='0':
            if self.lchild.isleaf:
                newnode=node(q, self)
                q=self.lchild.obj
                self.lchild=newnode
                self.insert(q[0], q[1], q[2])
            self.lchild.insert(hashcode, term, depth+1)
            return
        if self.rchild is None:
            newnode=node(q, self)
            self.rchild=newnode
        self.rchild.insert(hashcode, term, depth+1)
        return

class Btree(object):
    def __init__(self):
        self.root=node(None, None)

    def insert(self, hashcode, term):
        Node=self.root
        depth=0
        Node.insert(hashcode, term, depth)

    def delete(self, hashcode, term):
        depth=0
        Node=self.root
        for t in hashcode:
            depth+=1
            Node, _=( Node.ldown() if t=='0' else Node.rdown())
            if Node.lchild.isleaf:
                obj=Node.lchild.obj
                if len(obj[1])>1:
                    for index in range(len(obj[1])):
                        value=(np.equal(term, obj[1][index])).sum()
                        if int(value)==len(term):
                            del obj[1][index]
                            return
                    raise ValueError("The queried item not exist in this tree")
                value=(np.equal(term, obj[1][0])).sum()
                if int(value)==len(term):
                    Node.lchild=None
                    depth-=1
                    while depth>-1:
                        Node=Node.up()
                        t=hashcode[depth]
                        if t=='0':
                            if not( Node.rchild is None):
                                Node.lchild=None
                                return
                        else:
                            if not(Node.lchild is None):
                                Node.rchild=None
                                return
                        depth-=1
                    t=hashcode[0]
                    if t=='0':
                        Node.lchild=None
                        return
                    if t=='1':
                        Node.rchild=None
                        return
                else:
                    raise ValueError("The queried item not exist in this tree")

    def descend(self, hashcode):
        def desc(q, Node, depth):
            if Node.lchild.isleaf:
                return (Node.lchild, depth)
            t=q[depth]
            depth+=1
            if t=='0':
                newNode=Node.lchild
            elif t=='1':
                newNode=Node.rchild
            else:
                logger.error(f'error value of obj {t}')
            if newNode is None:
                return (Node, depth-1)
            Node=newNode
            return desc(q, Node, depth)
        Node, depth=desc(hashcode, self.root, 0)
        return (Node, depth)

    def descendants(self, Node, res):
        def descen(Node, res):
            if Node is None:
                return
            if not(Node.lchild is None) and Node.lchild.isleaf:
                res.append(Node.lchild)
                descen(Node.rchild, res)
                return
            descen(Node.lchild, res)
            descen(Node.rchild, res)
        descen(Node, res)

class LSHTree(object):
    def __init__(self, km):
        self.km=km
        self.generator=E2Family(50, k=self.km, w=30.0)
        self.func=self.generate_hashfunc()
        self.tree=Btree()

    def query(self, term):
        hashcode=self.list2vec(term)
        return self.tree.descent(hashcode)

    def add_batch(self, q_list, q_list_num):
        vec=np.array([self.list2vec(item) for item in q_list])
        #q_list=np.array(q_list)[np.argsort(vec)]
        #vec=np.sort(vec)
        for item, index in zip(vec, range(len(vec))):
            self._add_one(item, q_list_num[index])

    def add_one(self, image):
        hashcode=self.list2vec(image)
        self._add_one(hashcode, image)

    def delete_one(self, image):
        hashcode=self.list2vec(image)
        self.tree.delete(hashcode, image)

    def _add_one(self, hashcode, term):
        self.tree.insert(hashcode, term)

    def list2vec(self, term):
        code=np.array(self.func(term))
        code=code%2
        code=code.tolist()
        code=[str(item) for item in code]
        return ''.join(code)

    def generate_hashfunc(self):
        return self.generator.sample()

    def descend(self, image):
        hashcode=self.list2vec(image)
        return self.tree.descend(hashcode)

    def descendants(self, Node, res):
        return self.tree.descendants(Node, res)
        

class LSHForest(LSH):
    def __init__(self, l, km):
        self.km=km ##the maximum length
        self.l=l ## the number of trees
        self.forest=[LSHTree(self.km) for _ in range(self.l)]
        self.table=None

    def add_batch(self, q_list):   #build hash table
        if self.table is None:
            begin=0
            self.table=q_list
        else:
            begin=len(self.table)
            self.table.extend(q_list)
        q_list_num=[i for i in range(begin, len(self.table))]
        #print (q_list)
        [tree.add_batch(q_list, q_list_num) for tree in self.forest]
        logger.info("BUILDING TABLE SUC")

    def add_one(self, image):
        [tree.add_one(image) for tree in self.forest]

    def delete_one(self, image):
        [tree.delete_one(image) for tree in self.forest]


    def query(self, q, c=2, m=20):  # query from hash table
        descend=[tree.descend(q) for tree in self.forest]
        res=[]
        res_=[]

        x=max([item[1] for item in descend])
        logger.info(f"The value of x is {x}")
        while (x>0 and (len(res_)<c*self.l or len(res_)<m)):
            for i in range(self.l):
                if descend[i][1]==x:
                    self.forest[i].descendants(descend[i][0], res)
                    descend[i]=[descend[i][0].up(), descend[i][1]-1]
            x-=1
            res_=[]
            for Node in res:
                res_.extend(Node.obj[1])
            res_=list(set(res_))
        res_=[self.table[item] for item in res_]
        res_=np.array(res_)
        logger.info(f"FINISH QUERY, the number of candidate is {len(res_)}")
        
        return res_#[self.table[item] for item in res]

