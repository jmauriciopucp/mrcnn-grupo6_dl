# Segmentación de instancias con Mask R-CNN
 El siguiente repositorio fue construido por el Grupo #6 para el curso de Técnicas avanzadas de **Data Mining y Sistemas inteligentes** de la Maestría en Informática con mención en Ciencias de la Computación de la PUCP.
 
 Se tomó las fuentes propuestas por los autores principales del modelo [Mask R-CNN](https://arxiv.org/abs/1703.06870):
    
 - [Matterport, Inc](https://github.com/matterport/Mask_RCNN) - Construido bajo la versión 1.x de Tensorflow.
 - [Adam Kelly](https://github.com/akTwelve/Mask_RCNN) - Construido bajo la versión 2.x de Tensorflow.

 ## Configuración de VM para DeepLearning
 Debido a que en Google Colab no se contó con los recursos suficientes para entrenar el modelo Mask RCNN se creó una máquina virtual en Google Cloud Platform con las siguientes caracteristicas:

 - 30 GB de RAM
 - 8 vCPU
 - 1 GPU NVIDIA T4
 - 300 GB de Disco
 
 ## Configuración de ambiente
 Se deben seguir los siguientes pasos para configurar el ambiente de nuestro proyecto.
 1. Crear un entorno virtual utilizando la distribución de Anaconda, configurando la versión 3.7.7 de Python.
 2. Crear un kernel con nombre **maskrcnn**.
 3. Instalar la versión 2.1.0 de Tensorflow, utilizar la distribución de 'pip' para este propósito.
 4. Instalar la versión 2.1.0 de Tensorflow GPU y 10.1 del toolkit de CUDA(para propósitos de utilizar los controladores de NVIDIA). Utilizar la distribución de conda por ser más estable para este propósito.
 
 Instalar algunas dependencias adicionales detalladas en el cuadro de comandos de abajo.
 
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

 En la raíz de este repositorio se tiene los notebooks **training_model.ipynb** y **detection_model.py**, donde se realiza la experimentación del modelo Mask R-CNN (**ver imagen**) para la segementación de instancias de las imágenes del dataset D2S, imágenes tomadas desde un ángulo superior con la intención que el modelo sea una herramienta para la detección del número de productos y clase del producto en un cajero de supermercado, acelerando el procedimiento de compra de un producto.
<p align="center">
<img alt="mask-rcnn" src="https://user-images.githubusercontent.com/13455427/202597302-822b5d7c-5d9d-439d-b4f0-78d59e5ac8dd.PNG" width="534" height="240">
</p>


 ### Entrenamiento del modelo
 
 Para el entrenamiento de la red, se utilizó como base los pesos por defecto del modelo COCO y se buscó entrenar solo la última capa debido a restricciones de tiempo y memoria de los recursos utilizados. La red fue entrenada durante 40 épocas.
 
 ### Evaluación del modelo
 
 Se consideró el mAP como métrica de calidad del modelo y se utilizó Tensorboard 1.15.0 para visualizar el comportamiento de los *loss* del modelo. En la siguiente imagen se visualiza algunos de estos gráficos
 
 ![imagen1](https://user-images.githubusercontent.com/107210601/202592124-de7aa507-6d8a-4359-8cf7-870b6774bcb7.png)

### Detección de objetos

Se aplicó el modelo entrenado sobre la data de validación. A continuación se muestra la detección de objetos sobre una de estas imágenes.

![image](https://user-images.githubusercontent.com/107210601/202592606-248a783a-8f47-4a28-88c2-32d1ffbc53dd.png)

### Aplicación del modelo entrenado

El dataset D2S no provee las anotaciones que corresponden al conjunto de datos de testing. Por lo tanto, el grupo realizó la anotación manual de 30 imágenes utilizando la herramienta *VGG Image Annotator* disponible en http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.0.html.

Para realizar las anotaciones, se consultó la página https://universe.roboflow.com/nitt-vykgx/d2s-actual-split/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true que ayudó a identificar las clases de los objetos 
![image](https://user-images.githubusercontent.com/107210601/202593902-f03592e7-9827-47cf-bbc2-bbbbd974cae5.png)

Seguidamente, se graficó a mano alzada el polígono de cada imagen.

![image](https://user-images.githubusercontent.com/107210601/202593727-426d85ee-c5e8-4160-b4e4-b85c20993616.png)

![imagen 2](https://user-images.githubusercontent.com/107210601/202594548-eb9e36f9-6ffe-44e6-a174-7587da5e4f1e.png)


## Documentación de Experimentación
 Los notebooks fueron documentados usando el idioma español. El código fuente de Mask RCNN de **model.py** fue comentado para detallar los pasos (ver imagen) más importantes del algoritmo. Los comentarios fueron escritos en español y se pueden encontrar como:
 
        ############################################################
        ## Comentarios del G6:
        ############################################################

<p align="center">
<img src="https://user-images.githubusercontent.com/13455427/202596322-5d7c5b3d-8296-4325-ae00-893b47fc5664.PNG" data-canonical-src="https://user-images.githubusercontent.com/13455427/202596322-5d7c5b3d-8296-4325-ae00-893b47fc5664.PNG" width="280" height="230" />
</p>

 
 
 
