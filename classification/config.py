import concepts

example_name = {'SetFit/sst2': 'text', 'ag_news': 'text', 'yelp_polarity': 'text', 'dbpedia_14': 'content','Linhduongcute/SOF': 'text'}
concepts_from_labels = {'SetFit/sst2': ["negative","positive"], 'yelp_polarity': ["negative","positive"], 'ag_news': ["world", "sports", "business", "technology"], 'dbpedia_14': ["company","education","artist","athlete","office","transportation","building","natural","village","animal","plant","album","film","written"],'Linhduongcute/SOF': ["HQ","LQ_CLOSE","LQ_EDIT"]}
class_num = {'SetFit/sst2': 2, 'ag_news': 4, 'yelp_polarity': 2, 'dbpedia_14': 14,'Linhduongcute/SOF':3}

# Config for Roberta-Base baseline
finetune_epoch = {'SetFit/sst2': 3, 'ag_news': 2, 'yelp_polarity': 2, 'dbpedia_14': 2,'Linhduongcute/SOF': 2}
finetune_mlp_epoch = {'SetFit/sst2': 30, 'ag_news': 5, 'yelp_polarity': 3, 'dbpedia_14': 3,'Linhduongcute/SOF': 3}

# Config for CBM training
concept_set = {'SetFit/sst2': concepts.sst2, 'yelp_polarity': concepts.yelpp, 'ag_news': concepts.agnews, 'dbpedia_14': concepts.dbpedia,'Linhduongcute/SOF': concepts.SOF}
#cbl_epochs = {'SetFit/sst2': 30, 'ag_news': 5, 'yelp_polarity': 3, 'dbpedia_14': 3}
cbl_epochs = {'SetFit/sst2': 10, 'ag_news': 3, 'yelp_polarity': 2, 'dbpedia_14': 2,'Linhduongcute/SOF': 2}