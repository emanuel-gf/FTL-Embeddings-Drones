# %%
import fiftyone

print(fiftyone.list_datasets())



# %%
dataset = fiftyone.load_dataset('tiles5cm')
#fiftyone.delete_dataset('sao-bento-do-una-tiles-640')


# %%
fiftyone.launch_app(dataset, auto=False)
# %%

path_folder = "/Users/macuser/Documents/UN/projects/webinar-DL/data/tiles5cm/labelling"
dataset.draw_labels(
    output_dir=path_folder,
    label_fields=["painel-solar","caixa-dagua","caminhao","palmeira","telhado-barro"]

)
