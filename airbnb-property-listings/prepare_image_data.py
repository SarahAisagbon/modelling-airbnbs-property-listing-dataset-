
import glob
import numpy as np
import os
from PIL import Image

class PrepareImages:
    def __init__(self):
        self.imagepaths = []
        self.imageheights = []
        self.RGBimages = []
        self.resized_images = []
        
    def createFolder(self, path):
        '''
        This function is used to create a folder.
        
        Args:
            path: the string representation of the path for the new folder.
        '''
        try:
            if not os.path.exists(path):
                os.mkdir(path)
        except OSError:
            print ("Error: Creating directory. " +  path)
            pass
        
    def load_and_check_images(self):
        '''
        This function loads the images, checks if the images are RGB images and finds the height for each image.
        
        '''
        filepath = "/Users/sarahaisagbon/Documents/GitHub/Data-Science/airbnb-property-listings/images"
        #loop through the image folder and get the image for each subfolder
        self.imagepaths = glob.glob(filepath + '/**/*.png', recursive=True)
        
        #load each image
        for image in self.imagepaths:
            img = Image.open(image)
            img_arr = np.asarray(img)
            #first check if image is RGB, if not dicard
            if len(img_arr.shape)!=3:
                pass
            else:
                #create list of RGB image filepaths
                self.RGBimages.append(image)

                #get the height of each image and put it in a list
                height = img.height
                self.imageheights.append(height)

        print(f"There are {len(self.RGBimages)} RGB images")
    
    def resize_images(self):
        '''
        This function finds the minimum height alongst the images and resized all the RGB images.
        
        '''
        #find minimum height of all the images
        min_height = min(self.imageheights)
        
        for image in self.RGBimages:
            img = Image.open(image)
            height = img.height
            width = img.width
            #find the appropriate width for the new height
            new_width  = int(min_height * width / height)
            #resize all images to the same height and width as the smallest image
            new_image = img.resize((new_width, min_height))
            self.resized_images.append(new_image)
        print("All images resized!")
            
    def save_resized_image(self):
        '''
        This function creates processed_image folder and saved the resized image in the new folder.
        
        '''
        #create processed_images folder
        new_folder = "/Users/sarahaisagbon/Documents/GitHub/Data-Science/airbnb-property-listings/processed_images"
        PrepareImages.createFolder(self, new_folder)
        
        for old_img_path, img in zip(self.RGBimages, self.resized_images):
            imagepath = str(old_img_path).split("/")
            #save new version in processed_images folder
            new_imagepath = os.path.join(new_folder, imagepath[-1])
            img.save(new_imagepath)
        print("All resized images saved!")

    def processing(self):
        PrepareImages.load_and_check_images(self)
        PrepareImages.resize_images(self)
        PrepareImages.save_resized_image(self)

if __name__ == "__main__":
    resizing = PrepareImages()
    resizing.processing()
