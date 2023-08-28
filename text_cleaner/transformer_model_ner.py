from transformers import pipeline, Pipeline


class TransformerModelNer:
    def __init__(self, name: str, local_path: str, ent_name_to_detect: str) -> None:
        self._name = name
        self._local_path = local_path
        self._ent_name_to_detect = ent_name_to_detect
        
    def get_name(self) -> str:
        return self._name
    
    def get_local_path(self) -> str:
        return self._local_path
    
    def get_ent_name_to_detect(self) -> str:
        return self._ent_name_to_detect
    
    def get_pipe(self) -> Pipeline:
        try:
            pipe = pipeline('ner', model=self._local_path, grouped_entities=True)
        except:
            pipe = pipeline('ner', model=self._name, grouped_entities=True)
            
        return pipe