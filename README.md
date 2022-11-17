# Segmentación de instancias con Mask R-CNN
 El siguiente repositorio fue construido por el Grupo #6 para el curso de Técnicas avanzadas de **Data Mining y Sistemas inteligentes** de la Maestría en Informática con mención en Ciencias de la Computación de la PUCP.
 
 Se tomó las fuentes propuestas por los autores principales del modelo [Mask R-CNN](https://arxiv.org/abs/1703.06870):
    
 - [Matterport, Inc](https://github.com/matterport/Mask_RCNN) - Construido bajo la versión 1.x de Tensorflow.
 - [Adam Kelly](https://github.com/akTwelve/Mask_RCNN) - Construido bajo la versión 2.x de Tensorflow.
 
 ## Configuración de ambiente
 Se deben seguir los siguientes pasos para configurar el ambiente de nuestro proyecto.
 1. Crearemos nuestro entorno virtual utilizando la distribución de Anaconda, configurando la versión 3.7.7 de Python.
 2. Crearemos nuestro kernel con nombre **maskrcnn**.
 3. Instalaremos la versión 2.1.0 de Tensorflow, utilizaremos la distribución de 'pip' para este propósito.
 4. Instalaremos la versión 2.1.0 de Tensorflow GPU y 10.1 del toolkit de CUDA(para propósitos de utilizar los controladores de NVIDIA). Utilizaremos la distribución de conda por ser más estable para este propósito.
 
 Instalaremos algunas dependencias adicionales detalladas en el cuadro de comandos de abajo.
 
    $ conda create -n MaskRCNN anaconda python=3.7.7
    $ conda activate MaskRCNN
    $ conda install ipykernel
    $ python -m ipykernel install --user --name MaskRCNN --display-name "MaskRCNN"
    $ conda install tensorflow-gpu==2.1.0 cudatoolkit=10.1
    $ pip install tensorflow==2.1.0
    $ pip install jupyter
    $ pip install keras
    $ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all]
    
 ## Instalación de la librería Mask R-CNN
 Se debe ejecutar el script para la configuración de la librería **maskrcnn** la cual aloja los principales métodos para poder realizar la manipulación del modelo Mask R-CNN.
 Para ello, se deben ejecutar las siguientes líneas en el entorno virtual configurado líneas arriba:
 
    $ python setup.py install
    $ pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
 ## Dataset D2S  
 En la carpeta **dataset** se encuentran 4 carpetas con las imágenes y anotaciones de [MVTEC D2S](https://www.mvtec.com/company/research/datasets/mvtec-d2s):
 
 - **/train**: En esta carpeta se encuentran las 4380 imágenes para el entrenamiento de D2S.
 - **/val**: En esta carpeta se encuentran las 3600 imágenes para la validación de D2S.
 - **/test**: En esta carpeta se encuentran 30 imágenes para el test del modelo, tomadas como imágenes a ser inferidas por el modelo.
 - **/annotations**: En esta carpeta se encuentran las anotaciones para las imágenes del modelo, donde se tienen los archivos instances_train.json y instances_val.json .
 ## Experimentación con Mask R-CNN

 En la raíz de este repositorio tenemos los notebooks **training_model.ipynb** y **detection_model.py**, donde realizamos la experimentación del modelo Mask R-CNN para la segementación de instancias de las imágenes del dataset D2S, imágenes tomadas desde un ángulo superior con la intención que el modelo sea una herramienta para detección del número de productos y clase del producto en un cajero de supermercado, acelerando el procedimiento de compra de un producto.
 
 Muchas gracias.
