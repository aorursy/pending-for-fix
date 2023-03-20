#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from IPython.display import HTML
HTML('<center><iframe width="560" height="315" src="https://www.youtube.com/embed/AfK9LPNj-Zo" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')


# In[ ]:


import os
from os import listdir
import pandas as pd
import numpy as np
import glob
import tqdm
from typing import Dict
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#plotly
get_ipython().system('pip install chart_studio')
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

#color
from colorama import Fore, Back, Style

import seaborn as sns
sns.set(style="whitegrid")

#pydicom
import pydicom

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# Settings for pretty nice plots
plt.style.use('fivethirtyeight')
plt.show()


# In[ ]:


# List files available
list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))


# In[ ]:


IMAGE_PATH = "../input/osic-pulmonary-fibrosis-progressiont/"

train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

print(Fore.YELLOW + 'Training data shape: ',Style.RESET_ALL,train_df.shape)
train_df.head(5)


# In[ ]:


train_df.groupby(['SmokingStatus']).count()['Sex'].to_frame()


# In[ ]:


# Null values and Data types
print(Fore.YELLOW + 'Train Set !!',Style.RESET_ALL)
print(train_df.info())
print('-------------')
print(Fore.BLUE + 'Test Set !!',Style.RESET_ALL)
print(test_df.info())


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


# Total number of Patient in the dataset(train+test)

print(Fore.YELLOW +"Total Patients in Train set: ",Style.RESET_ALL,train_df['Patient'].count())
print(Fore.BLUE +"Total Patients in Test set: ",Style.RESET_ALL,test_df['Patient'].count())


# In[ ]:


print(Fore.YELLOW + "The total patient ids are",Style.RESET_ALL,f"{train_df['Patient'].count()},", Fore.BLUE + "from those the unique ids are", Style.RESET_ALL, f"{train_df['Patient'].value_counts().shape[0]}.")


# In[ ]:


train_patient_ids = set(train_df['Patient'].unique())
test_patient_ids = set(test_df['Patient'].unique())

train_patient_ids.intersection(test_patient_ids)


# In[ ]:


columns = train_df.keys()
columns = list(columns)
print(columns)


# In[ ]:


train_df['Patient'].value_counts().max()


# In[ ]:


test_df['Patient'].value_counts().max()


# In[ ]:


np.quantile(train_df['Patient'].value_counts(), 0.75) - np.quantile(test_df['Patient'].value_counts(), 0.25)


# In[ ]:


print(np.quantile(train_df['Patient'].value_counts(), 0.95))
print(np.quantile(test_df['Patient'].value_counts(), 0.95))


# In[ ]:


files = folders = 0

path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"

for _, dirnames, filenames in os.walk(path):
  # ^ this idiom means "we won't be using this value"
    files += len(filenames)
    folders += len(dirnames)
#print(Fore.YELLOW +"Total Patients in Train set: ",Style.RESET_ALL,train_df['Patient'].count())
print(Fore.YELLOW +f'{files:,}',Style.RESET_ALL,"files/images, " + Fore.BLUE + f'{folders:,}',Style.RESET_ALL ,'folders/patients')


# In[ ]:


files = []
for _, dirnames, filenames in os.walk(path):
  # ^ this idiom means "we won't be using this value"
    files.append(len(filenames))

print(Fore.YELLOW +f'{round(np.mean(files)):,}',Style.RESET_ALL,'average files/images per patient')
print(Fore.BLUE +f'{round(np.max(files)):,}',Style.RESET_ALL, 'max files/images per patient')
print(Fore.GREEN +f'{round(np.min(files)):,}',Style.RESET_ALL,'min files/images per patient')


# In[ ]:


patient_df = train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates()
patient_df.head()


# In[ ]:


# Creating unique patient lists and their properties. 
train_dir = '../input/osic-pulmonary-fibrosis-progression/train/'
test_dir = '../input/osic-pulmonary-fibrosis-progression/test/'

patient_ids = os.listdir(train_dir)
patient_ids = sorted(patient_ids)

#Creating new rows
no_of_instances = []
age = []
sex = []
smoking_status = []

for patient_id in patient_ids:
    patient_info = train_df[train_df['Patient'] == patient_id].reset_index()
    no_of_instances.append(len(os.listdir(train_dir + patient_id)))
    age.append(patient_info['Age'][0])
    sex.append(patient_info['Sex'][0])
    smoking_status.append(patient_info['SmokingStatus'][0])

#Creating the dataframe for the patient info    
patient_df = pd.DataFrame(list(zip(patient_ids, no_of_instances, age, sex, smoking_status)), 
                                 columns =['Patient', 'no_of_instances', 'Age', 'Sex', 'SmokingStatus'])
print(patient_df.info())
patient_df.head()


# In[ ]:


patient_df['SmokingStatus'].value_counts()


# In[ ]:


patient_df['SmokingStatus'].value_counts().iplot(kind='bar',
                                              yTitle='Counts', 
                                              linecolor='black', 
                                              opacity=0.7,
                                              color='blue',
                                              theme='pearl',
                                              bargap=0.5,
                                              gridcolor='white',
                                              title='Distribution of the SmokingStatus column in the Unique Patient Set')


# In[ ]:


train_df['Weeks'].value_counts().head()


# In[ ]:


train_df['Weeks'].value_counts().iplot(kind='barh',
                                      xTitle='Counts(Weeks)', 
                                      linecolor='black', 
                                      opacity=0.7,
                                      color='#FB8072',
                                      theme='pearl',
                                      bargap=0.2,
                                      gridcolor='white',
                                      title='Distribution of the Weeks in the training set')


# In[ ]:


train_df['Weeks'].iplot(kind='hist',
                              xTitle='Weeks', 
                              yTitle='Counts',
                              linecolor='black', 
                              opacity=0.7,
                              color='#FB8072',
                              theme='pearl',
                              bargap=0.2,
                              gridcolor='white',
                              title='Distribution of the Weeks in the training set')


# In[ ]:


fig = px.scatter(train_df, x="Weeks", y="Age", color='Sex')
fig.show()


# In[ ]:


train_df['FVC'].value_counts()


# In[ ]:


train_df['FVC'].iplot(kind='hist',
                      xTitle='Lung Capacity(ml)', 
                      linecolor='black', 
                      opacity=0.8,
                      color='#FB8072',
                      bargap=0.5,
                      gridcolor='white',
                      title='Distribution of the FVC in the training set')


# In[ ]:


fig = px.scatter(train_df, x="FVC", y="Percent", color='Age')
fig.show()


# In[ ]:


FVC seems to related Percent linearly. Makes sense as both terms are proportional.


# In[ ]:


fig = px.scatter(train_df, x="FVC", y="Age", color='Sex')
fig.show()


# In[ ]:


fig = px.scatter(train_df, x="FVC", y="Weeks", color='SmokingStatus')
fig.show()


# In[ ]:


patient = train_df[train_df.Patient == 'ID00228637202259965313869']
fig = px.line(patient, x="Weeks", y="FVC", color='SmokingStatus')
fig.show()


# In[ ]:


train_df['Percent'].value_counts()


# In[ ]:


train_df['Percent'].iplot(kind='hist',bins=30,color='blue',xTitle='Percent distribution',yTitle='Count')


# In[ ]:


df = train_df
fig = px.violin(df, y='Percent', x='SmokingStatus', box=True, color='Sex', points="all",
          hover_data=train_df.columns)
fig.show()


# In[ ]:


plt.figure(figsize=(16, 6))
ax = sns.violinplot(x = train_df['SmokingStatus'], y = train_df['Percent'], palette = 'Reds')
ax.set_xlabel(xlabel = 'Smoking Habit', fontsize = 15)
ax.set_ylabel(ylabel = 'Percent', fontsize = 15)
ax.set_title(label = 'Distribution of Smoking Status Over Percentage', fontsize = 20)
plt.show()


# In[ ]:


fig = px.scatter(train_df, x="Age", y="Percent", color='SmokingStatus')
fig.show()


# In[ ]:


patient = train_df[train_df.Patient == 'ID00228637202259965313869']
fig = px.line(patient, x="Weeks", y="Percent", color='SmokingStatus')
fig.show()


# In[ ]:


patient = train_df[train_df.Patient == 'ID00228637202259965313869']
fig = px.line(patient, x="Weeks", y="Percent", color='SmokingStatus')
fig.show()


# In[ ]:


patient_df['Age'].iplot(kind='hist',bins=30,color='red',xTitle='Ages of distribution',yTitle='Count')


# In[ ]:


patient_df['SmokingStatus'].value_counts()


# In[ ]:


plt.figure(figsize=(16, 6))
sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)
sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)
sns.kdeplot(patient_df.loc[patient_df['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes', shade=True)

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');


# In[ ]:


plt.figure(figsize=(16, 6))
ax = sns.violinplot(x = patient_df['SmokingStatus'], y = patient_df['Age'], palette = 'Reds')
ax.set_xlabel(xlabel = 'Smoking habit', fontsize = 15)
ax.set_ylabel(ylabel = 'Age', fontsize = 15)
ax.set_title(label = 'Distribution of Smokers over Age', fontsize = 20)
plt.show()


# In[ ]:


plt.figure(figsize=(16, 6))
sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)
sns.kdeplot(patient_df.loc[patient_df['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');


# In[ ]:


patient_df['Sex'].value_counts()


# In[ ]:


patient_df['Sex'].value_counts().iplot(kind='bar',
                                          yTitle='Count', 
                                          linecolor='black', 
                                          opacity=0.7,
                                          color='blue',
                                          theme='pearl',
                                          bargap=0.8,
                                          gridcolor='white',
                                          title='Distribution of the Sex column in Patient Dataframe')


# In[ ]:


plt.figure(figsize=(16, 6))
a = sns.countplot(data=patient_df, x='SmokingStatus', hue='Sex')

for p in a.patches:
    a.annotate(format(p.get_height(), ','), 
           (p.get_x() + p.get_width() / 2., 
            p.get_height()), ha = 'center', va = 'center', 
           xytext = (0, 4), textcoords = 'offset points')

plt.title('Gender split by SmokingStatus', fontsize=16)
sns.despine(left=True, bottom=True);


# In[ ]:


fig = px.box(patient_df, x="Sex", y="Age", points="all")
fig.show()


# In[ ]:


# Extract patient id's for the training set
ids_train = train_df.Patient.values
# Extract patient id's for the validation set
ids_test = test_df.Patient.values
#print(Fore.YELLOW +"Total Patients in Train set: ",Style.RESET_ALL,train_df['Patient'].count())
# Create a "set" datastructure of the training set id's to identify unique id's
ids_train_set = set(ids_train)
print(Fore.YELLOW + "There are",Style.RESET_ALL,f'{len(ids_train_set)}', Fore.BLUE + 'unique Patient IDs',Style.RESET_ALL,'in the training set')
# Create a "set" datastructure of the validation set id's to identify unique id's
ids_test_set = set(ids_test)
print(Fore.YELLOW + "There are", Style.RESET_ALL, f'{len(ids_test_set)}', Fore.BLUE + 'unique Patient IDs',Style.RESET_ALL,'in the test set')

# Identify patient overlap by looking at the intersection between the sets
patient_overlap = list(ids_train_set.intersection(ids_test_set))
n_overlap = len(patient_overlap)
print(Fore.YELLOW + "There are", Style.RESET_ALL, f'{n_overlap}', Fore.BLUE + 'Patient IDs',Style.RESET_ALL, 'in both the training and test sets')
print('')
print(Fore.CYAN + 'These patients are in both the training and test datasets:', Style.RESET_ALL)
print(f'{patient_overlap}')


# In[ ]:


corrmat = train_df.corr() 
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap = 'RdYlBu_r', linewidths = 0.5) 


# In[ ]:


# 6. <a id='visual'>Visualising Images : DECOM 🗺️</a>  


# In[ ]:


print(Fore.YELLOW + 'Train .dcm number of images:',Style.RESET_ALL, len(list(os.listdir('../input/osic-pulmonary-fibrosis-progression/train'))), '\n' +
      Fore.BLUE + 'Test .dcm number of images:',Style.RESET_ALL, len(list(os.listdir('../input/osic-pulmonary-fibrosis-progression/test'))), '\n' +
      '--------------------------------', '\n' +
      'There is the same number of images as in train/ test .csv datasets')


# In[ ]:


def plot_pixel_array(dataset, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.grid(False)
    plt.imshow(dataset.pixel_array, cmap='gray') # cmap=plt.cm.bone)
    plt.show()


# In[ ]:


# https://www.kaggle.com/schlerp/getting-to-know-dicom-and-the-data
def show_dcm_info(dataset):
    print(Fore.YELLOW + "Filename.........:",Style.RESET_ALL,file_path)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print(Fore.BLUE + "Patient's name......:",Style.RESET_ALL, display_name)
    print(Fore.BLUE + "Patient id..........:",Style.RESET_ALL, dataset.PatientID)
    print(Fore.BLUE + "Patient's Sex.......:",Style.RESET_ALL, dataset.PatientSex)
    print(Fore.YELLOW + "Modality............:",Style.RESET_ALL, dataset.Modality)
    print(Fore.GREEN + "Body Part Examined..:",Style.RESET_ALL, dataset.BodyPartExamined)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print(Fore.BLUE + "Image size.......:",Style.RESET_ALL," {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print(Fore.YELLOW + "Pixel spacing....:",Style.RESET_ALL,dataset.PixelSpacing)
            dataset.PixelSpacing = [1, 1]
        plt.figure(figsize=(10, 10))
        plt.imshow(dataset.pixel_array, cmap='gray')
        plt.show()
for file_path in glob.glob('../input/osic-pulmonary-fibrosis-progression/train/*/*.dcm'):
    dataset = pydicom.dcmread(file_path)
    show_dcm_info(dataset)
    break # Comment this out to see all


# In[ ]:


imdir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140"
print("total images for patient ID00123637202217151272140: ", len(os.listdir(imdir)))

# view first (columns*rows) images in order
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 5
imglist = os.listdir(imdir)
for i in range(1, columns*rows +1):
    filename = imdir + "/" + str(i) + ".dcm"
    ds = pydicom.dcmread(filename)
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap='gray')
plt.show()


# In[ ]:


imdir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140"
print("total images for patient ID00123637202217151272140: ", len(os.listdir(imdir)))

# view first (columns*rows) images in order
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 5
imglist = os.listdir(imdir)
for i in range(1, columns*rows +1):
    filename = imdir + "/" + str(i) + ".dcm"
    ds = pydicom.dcmread(filename)
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap='jet')
plt.show()


# In[ ]:


apply_resample = False

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


# In[ ]:


def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


# In[ ]:


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# In[ ]:


def set_lungwin(img, hu=[-1200., 600.]):
    lungwin = np.array(hu)
    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


# In[ ]:


scans = load_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')
scan_array = set_lungwin(get_pixels_hu(scans))


# In[ ]:


# Resample to 1mm (An optional step, it may not be relevant to this competition because of the large slice thickness on the z axis)

from scipy.ndimage.interpolation import zoom

def resample(imgs, spacing, new_spacing):
    new_shape = np.round(imgs.shape * spacing / new_spacing)
    true_spacing = spacing * imgs.shape / new_shape
    resize_factor = new_shape / imgs.shape
    imgs = zoom(imgs, resize_factor, mode='nearest')
    return imgs, true_spacing, new_shape

spacing_z = (scans[-1].ImagePositionPatient[2] - scans[0].ImagePositionPatient[2]) / len(scans)

if apply_resample:
    scan_array_resample = resample(scan_array, np.array(np.array([spacing_z, *scans[0].PixelSpacing])), np.array([1.,1.,1.]))[0]


# In[ ]:


import imageio
from IPython.display import Image

imageio.mimsave("/tmp/gif.gif", scan_array, duration=0.0001)
Image(filename="/tmp/gif.gif", format='png')


# In[ ]:


import matplotlib.animation as animation

fig = plt.figure()

ims = []
for image in scan_array:
    im = plt.imshow(image, animated=True, cmap="Greys")
    plt.axis("off")
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                repeat_delay=1000)


# In[ ]:


HTML(ani.to_jshtml())


# In[ ]:


HTML(ani.to_html5_video())


# In[ ]:


def extract_dicom_meta_data(filename: str) -> Dict:
    # Load image
    
    image_data = pydicom.read_file(filename)
    img=np.array(image_data.pixel_array).flatten()
    row = {
        'Patient': image_data.PatientID,
        'body_part_examined': image_data.BodyPartExamined,
        'image_position_patient': image_data.ImagePositionPatient,
        'image_orientation_patient': image_data.ImageOrientationPatient,
        'photometric_interpretation': image_data.PhotometricInterpretation,
        'rows': image_data.Rows,
        'columns': image_data.Columns,
        'pixel_spacing': image_data.PixelSpacing,
        'window_center': image_data.WindowCenter,
        'window_width': image_data.WindowWidth,
        'modality': image_data.Modality,
        'StudyInstanceUID': image_data.StudyInstanceUID,
        'SeriesInstanceUID': image_data.StudyInstanceUID,
        'StudyID': image_data.StudyInstanceUID, 
        'SamplesPerPixel': image_data.SamplesPerPixel,
        'BitsAllocated': image_data.BitsAllocated,
        'BitsStored': image_data.BitsStored,
        'HighBit': image_data.HighBit,
        'PixelRepresentation': image_data.PixelRepresentation,
        'RescaleIntercept': image_data.RescaleIntercept,
        'RescaleSlope': image_data.RescaleSlope,
        'img_min': np.min(img),
        'img_max': np.max(img),
        'img_mean': np.mean(img),
        'img_std': np.std(img)}

    return row


# In[ ]:


train_image_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/train'
train_image_files = glob.glob(os.path.join(train_image_path, '*', '*.dcm'))

meta_data_df = []
for filename in tqdm.tqdm(train_image_files):
    try:
        meta_data_df.append(extract_dicom_meta_data(filename))
    except Exception as e:
        continue


# In[ ]:


# Convert to a pd.DataFrame from dict
meta_data_df = pd.DataFrame.from_dict(meta_data_df)
meta_data_df.head()


# In[ ]:


# source: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154658
folder='train'
PATH='../input/osic-pulmonary-fibrosis-progression/'

last_index = 2

column_names = ['image_name', 'dcm_ImageOrientationPatient', 
                'dcm_ImagePositionPatient', 'dcm_PatientID',
                'dcm_PatientName', 'dcm_PatientSex'
                'dcm_rows', 'dcm_columns']

def extract_DICOM_attributes(folder):
    patients_folder = list(os.listdir(os.path.join(PATH, folder)))
    df = pd.DataFrame()
    
    i = 0
    
    for patient_id in patients_folder:
   
        img_path = os.path.join(PATH, folder, patient_id)
        
        print(img_path)
        
        images = list(os.listdir(img_path))
        
        #df = pd.DataFrame()

        for image in images:
            image_name = image.split(".")[0]

            dicom_file_path = os.path.join(img_path,image)
            dicom_file_dataset = pydicom.read_file(dicom_file_path)
                
            '''
            print(dicom_file_dataset.dir("pat"))
            print(dicom_file_dataset.data_element("ImageOrientationPatient"))
            print(dicom_file_dataset.data_element("ImagePositionPatient"))
            print(dicom_file_dataset.data_element("PatientID"))
            print(dicom_file_dataset.data_element("PatientName"))
            print(dicom_file_dataset.data_element("PatientSex"))
            '''
            
            imageOrientationPatient = dicom_file_dataset.ImageOrientationPatient
            #imagePositionPatient = dicom_file_dataset.ImagePositionPatient
            patientID = dicom_file_dataset.PatientID
            patientName = dicom_file_dataset.PatientName
            patientSex = dicom_file_dataset.PatientSex
        
            rows = dicom_file_dataset.Rows
            cols = dicom_file_dataset.Columns
            
            #print(rows)
            #print(columns)
            
            temp_dict = {'image_name': image_name, 
                                    'dcm_ImageOrientationPatient': imageOrientationPatient,
                                    #'dcm_ImagePositionPatient':imagePositionPatient,
                                    'dcm_PatientID': patientID, 
                                    'dcm_PatientName': patientName,
                                    'dcm_PatientSex': patientSex,
                                    'dcm_rows': rows,
                                    'dcm_columns': cols}


            df = df.append([temp_dict])
            
        i += 1
        
        if i == last_index:
            break
            
    return df


# In[ ]:


extract_DICOM_attributes('train')


# In[ ]:


import pandas_profiling as pdp


# In[ ]:


train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


# In[ ]:


profile_train_df = pdp.ProfileReport(train_df)


# In[ ]:


profile_train_df


# In[ ]:


profile_test_df = pdp.ProfileReport(test_df)


# In[ ]:


profile_test_df

