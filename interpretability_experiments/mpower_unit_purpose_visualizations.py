# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Attribution(object):
    def __init__(self):
        self.data = {}
    
    def update_unit(self, layer, unit, row):
        if layer not in self.data:
            self.data.update({layer:{}})
        
        if unit not in self.data[layer]:
            self.data[layer].update({unit:pd.DataFrame(columns = ['perodic', 'peaks', 'spikes', 'amplitude'])})
            
        self.data[layer][unit].append(row, ignore_index=True)
        
    def get_important_feature(self, layer, unit):
        max_value = float("-inf")
        max_outcome = ""
        
        for col in self.data[layer][unit]:
            curr_count = col.max()
            outcome = col.idxmax()
            if curr_count > max_value and outcome:
                max_value = curr_count
                max_outcome = col
        
        return outcome

    
    def visualize_model(self, layers):
        result = []
        
        # detectors = {"perodic":"Shape","peaks":"Shape","spikes":"Shape","amplitude":"Intensity"}
        
        for i in layers:
            for unit in self.data[i]:
                result.append(self.get_important_feature(i,unit))
                
        
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))


        data = [float(x.split()[0]) for x in result]
        ingredients = [x.split()[-1] for x in result]


        def func(pct, allvals):
            absolute = int(pct/100.*np.sum(allvals))
            return "{:.1f}%\n({:d} g)".format(pct, absolute)


        wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                        textprops=dict(color="w"))

        ax.legend(wedges, ingredients,
                title="Detector Type",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=8, weight="bold")

        ax.set_title("Model Detectors Visualization")

        plt.show()        
                
        