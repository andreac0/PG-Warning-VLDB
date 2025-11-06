import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import lance
import pyarrow as pa
from FlagEmbedding import BGEM3FlagModel
      
def average_pool(last_hidden_states: Tensor,attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5Embedding:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
     
    def compute(self, input_texts, N = 50):

        emb = []
        i = 0
        for j in tqdm(range(1,-(len(input_texts)//-N)+1)):
            
            # Tokenize the input texts
            batch_dict = self.tokenizer(input_texts[i:j*N], max_length=512, padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # normalize embeddings
            emb.extend(embeddings.tolist())
            i = i+N
        
        return emb
    

class E5EmbeddingCUDA:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').cuda()
        
    def compute(self, input_texts, N=40, db_storage = False, name='dataset.lance'):

        # if db_storage:
        #     # duck db
        #     num_columns = 1024
        #     column_names = [f"col_{i}" for i in range(1, num_columns + 1)]
        #     column_definitions = ", ".join([f"{col} FLOAT" for col in column_names])

        #     # Create the table with 1024 columns
        #     table_name = "embeddings"
        #     create_table_query = f"""
        #         CREATE TABLE IF NOT EXISTS {table_name} (
        #             {column_definitions}
        #         )
        #     """
        #     # Insert the computed data into DuckDB
        #     insert_query = f"""
        #         INSERT INTO embeddings 
        #         VALUES ({', '.join(['?'] * 1024)})
        #     """
        #     db_conn.execute(create_table_query)

        emb = []
        i = 0
        for j in tqdm(range(1, -(len(input_texts) // -N) + 1)):
            # Tokenize the input texts
            torch.cuda.empty_cache()

            batch_dict = self.tokenizer(
                input_texts[i:j * N],
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            # Move tensors to CUDA
            batch_dict = {key: tensor.cuda() for key, tensor in batch_dict.items()}

            outputs = self.model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # Normalize embeddings
            if db_storage == False:
                emb.extend(embeddings.cpu().tolist())  # Move back to CPU for storage
            else:
                # # Insert data row by row
                # for row in embeddings.cpu().tolist():
                #     db_conn.execute(insert_query, row)
                if i == 0:
                    tbl = pa.Table.from_arrays([input_texts[i:j * N],pa.array(embeddings.cpu().tolist())], names=["card","value"])
                    lance.write_dataset(tbl, name, mode= "create")
                else:
                    tbl = pa.Table.from_arrays([input_texts[i:j * N],pa.array(embeddings.cpu().tolist())], names=["card","value"])
                    lance.write_dataset(tbl, name, mode= "append")
            i += N

        return emb


class BGEEmbeddingCUDA:
    def __init__(self):
        self.model = BGEM3FlagModel('BAAI/bge-m3')
    
    def compute(self, input_texts, N=40, db_storage = False, name='dataset.lance'):

        emb = []
        i = 0
        for j in tqdm(range(1, -(len(input_texts) // -N) + 1)):

            if db_storage == False:
                emb.extend(self.model.encode(input_texts[i:j * N], 
                                            batch_size=12, 
                                            max_length=4096, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                            )['dense_vecs'])
            else:
                if i == 0:
                    tbl = pa.Table.from_arrays([input_texts[i:j * N],pa.array(self.model.encode(input_texts[i:j * N], 
                                            batch_size=12, 
                                            max_length=4096, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                            )['dense_vecs'].tolist())], names=["value"])
                    lance.write_dataset(tbl, name, mode= "create")
                else:
                    tbl = pa.Table.from_arrays([input_texts[i:j * N],pa.array(self.model.encode(input_texts[i:j * N], 
                                            batch_size=12, 
                                            max_length=4096, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                            )['dense_vecs'].tolist())], names=["value"])
                    lance.write_dataset(tbl, name, mode= "append")

            i += N
        
        return emb