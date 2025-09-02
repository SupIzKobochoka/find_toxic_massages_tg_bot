import pandas as pd
from pandarallel import pandarallel
from pymorphy3 import MorphAnalyzer   

pandarallel.initialize(nb_workers=12, progress_bar=True)

def get_normal_series(se: pd.Series) -> pd.Series:  
    morph = MorphAnalyzer()
    def get_normal_list(li: list[str]) -> list[str]:
        return [morph.parse(i)[0].normal_form for i in li]

    return se.parallel_apply(get_normal_list)

if __name__ == '__main__':
    se = pd.Series([
        ['мама', 'папа'],
        ["мамочки", 'папочки'],
        ['огромнейший']
                   ])

    print(get_normal_series(se))
            