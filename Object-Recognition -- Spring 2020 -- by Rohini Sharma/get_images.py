import os






directory = "/Users/rohinisharma/Projects/MURIproj/"


def rename_imgs(dir):
    for image in os.listdir(dir): 
        current = dir + image
        new_name = dir + image.split("_")[0] + "_segmented.png"
        #print(new_name)
        os.rename(current, new_name)


    

final_dir = directory +  "Bear_Without_Background/"
rename_imgs(final_dir)