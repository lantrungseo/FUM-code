from keras.utils import Sequence
import numpy as np 
import dask.dataframe as dd
import json

class NewsFetcher():
    def __init__(self,news_title,news_content,news_vert,news_subvert,news_entity):
        self.news_title = news_title
        self.news_content = news_content
        self.news_vert = news_vert
        self.news_entity = news_entity
        self.news_subvert = news_subvert
        
    def fetch(self,docids):
        bz,n = docids.shape # bz = end - start + 1, n = number of news (could be npratio+1=5, or MAX_CLICK=50)
        #print(docids.shape)
        news_title = self.news_title[docids] #(bz, n, MAX_TITLE)
        # print('--news_title--')
        # print(self.news_title.shape)
        # print(news_title.shape)
        news_content = self.news_content[docids] # (bz, n, MAX_CONTENT)
        # print('--news_content--')
        # print(news_content)
        news_vert = self.news_vert[docids].reshape((bz,n,1)) # (bz, n, 1)
        # print('--news_vert--')
        # print(news_vert)
        news_subvert = self.news_subvert[docids].reshape((bz,n,1)) # (bz, n, 1)
        # print('--news_subvert--')
        # print(news_subvert)
        news_entity = self.news_entity[docids] # (bz, n, MAX_ENTITY)
        # print(news_entity.shape)
        news_info = np.concatenate([news_title,news_vert,news_subvert,news_content,news_entity,],axis=-1)
        
        return news_info # (bz, n, MAX_TITLE+MAX_CONTENT+MAX_ENTITY+2)
    
    def fetch_dim1(self,docids):
        bz, = docids.shape
        news_title = self.news_title[docids] #(bz, MAX_TITLE)
        news_content = self.news_content[docids] # (bz, MAX_CONTENT)
        news_vert = self.news_vert[docids].reshape((bz,1)) # (bz, 1)
        news_subvert = self.news_subvert[docids].reshape((bz,1)) # (bz, 1)
        news_entity = self.news_entity[docids] # (bz, MAX_ENTITY)
        news_info = np.concatenate([news_title,news_vert,news_subvert,news_content,news_entity],axis=-1)
        
        return news_info # (bz, MAX_TITLE+MAX_CONTENT+MAX_ENTITY+2)

class get_train_generator_2(Sequence):
    def __init__(self,news_fetcher,clicked_news,user_id, news_id, label, batch_size):
        self.news_fetcher = news_fetcher
        self.clicked_news = clicked_news

        self.user_id = user_id
        self.doc_id = news_id
        self.label = label
        
        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]
    
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        label = self.label[start:ed]
        
        doc_ids = self.doc_id[start:ed]
        info= self.news_fetcher.fetch(doc_ids)
        
        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]
        user_info = self.news_fetcher.fetch(clicked_ids)
        
        click_mask = clicked_ids>0
        click_mask = np.array(click_mask,dtype='float32')

        return ([info, user_info],label)

class get_hir_train_generator(Sequence):
    def __init__(self,news_fetcher,clicked_news_path, news_id_path, label_path, batch_size, train_meta_path):
        self.news_fetcher = news_fetcher

        self.clicked_news_reader = dd.read_csv(clicked_news_path,header=None)
        self.doc_id_reader =  dd.read_csv(news_id_path,header=None)
        self.label_reader =  dd.read_csv(label_path,header=None)
        # add partition metadata for clicked_news csv file
        npart = self.clicked_news_reader.npartitions
        self.clicked_news_ranges_by_part = []
        for i in range(npart):
            partI = self.clicked_news_reader.get_partition(i).compute().values
            if len(self.clicked_news_ranges_by_part) == 0:
                self.clicked_news_ranges_by_part.append([0, len(partI)])
            else:
                self.clicked_news_ranges_by_part.append([self.clicked_news_ranges_by_part[-1][1], self.clicked_news_ranges_by_part[-1][1]+len(partI)])

        with open(train_meta_path, 'r') as f:
            self.train_meta = json.load(f)
        self.batch_size = batch_size
        self.ImpNum = self.train_meta["impressionNum"]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
    def get_clicked_sessions(self, start, ed):
        start_part_i = None
        end_part_i = None
        for i in range(len(self.clicked_news_ranges_by_part)):
            if start_part_i == None and start >= self.clicked_news_ranges_by_part[i][0] and start < self.clicked_news_ranges_by_part[i][1]:
                start_part_i = i
            if end_part_i == None and ed >= self.clicked_news_ranges_by_part[i][0] and ed < self.clicked_news_ranges_by_part[i][1]:
                end_part_i = i  
        part_start, part_end = start - self.clicked_news_ranges_by_part[start_part_i][0], ed - self.clicked_news_ranges_by_part[end_part_i][0]
        if start_part_i == end_part_i:
            return self.clicked_news_reader.get_partition(start_part_i).loc[part_start:part_end].compute().values
        else:
            half_start = self.clicked_news_reader.get_partition(start_part_i).loc[part_start:].compute().value
            half_end = self.clicked_news_reader.get_partition(end_part_i).loc[:part_end].compute().values
            return np.concatenate([
                half_start,
                half_end
            ], axis=0)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size - 1
        if ed>= self.ImpNum:
            ed = self.ImpNum-1
        label = self.label_reader.loc[start:ed].compute().values
        
        doc_ids = self.doc_id_reader.loc[start:ed].compute().values

        info= self.news_fetcher.fetch(doc_ids)
        
        clicked_ids = self.get_clicked_sessions(start, ed)
        user_info = self.news_fetcher.fetch(clicked_ids)
        
        # click_mask = clicked_ids>0
        # click_mask = np.array(click_mask,dtype='float32')

        #return ([info, user_info],[label])
        return ([info, user_info],label) # should be like this


class get_hir_user_generator(Sequence):
    def __init__(self,news_fetcher,clicked_news,batch_size):
        self.news_fetcher = news_fetcher

        self.clicked_news = clicked_news

        self.batch_size = batch_size
        self.ImpNum = self.clicked_news.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
            
        clicked_ids = self.clicked_news[start:ed]
        user_info = self.news_fetcher.fetch(clicked_ids)
        
        return user_info


class get_hir_news_generator(Sequence):
    def __init__(self,news_fetcher,batch_size):
        self.news_fetcher = news_fetcher

        self.batch_size = batch_size
        self.ImpNum = news_fetcher.news_title.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    
    def __get_news(self,docids):
        news_emb = self.news_emb[docids]

        return news_emb
    
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
            
        docids = np.array([i for i in range(start,ed)])
            
        info = self.news_fetcher.fetch_dim1(docids)

        return info