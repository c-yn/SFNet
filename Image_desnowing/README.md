### Download the Datasets
- SRRS [[gdrive](https://drive.google.com/file/d/11h1cZ0NXx6ev35cl5NKOAL3PCgLlWUl2/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1VXqsamkl12fPsI1Qek97TQ?pwd=vcfg)]
- CSD [[gdrive](https://drive.google.com/file/d/1pns-7uWy-0SamxjA40qOCkkhSu7o7ULb/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1N52Jnx0co9udJeYrbd3blA?pwd=sb4a)]
- Snow100K [[gdrive](https://drive.google.com/file/d/19zJs0cJ6F3G3IlDHLU2BO7nHnCTMNrIS/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1QGd5z9uM6vBKPnD5d7jQmA?pwd=aph4)]

### Training

~~~
python main.py --data CSD --mode train --data_dir your_path/CSD
python main.py --data SRRS --mode train --data_dir your_path/SRRS
python main.py --data Snow100K --mode train --data_dir your_path/Snow100K
~~~

### Evaluation
#### Download the model
- CSD [gdrive, [Baidu](https://pan.baidu.com/s/1q8Mp7iUoH-7-3PYnEZJQzw?pwd=86wc)]
- SRRS [[gdrive](https://drive.google.com/file/d/1MDDwPH0_MNNWT4YoyGrCXxpvE9LGQ9VM/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1K1mra1fGYnjEonpV9URBCg?pwd=sl5k)]
- Snow100K [[gdrive](https://drive.google.com/file/d/14S4JtFlw7zV0k9m0ka3NTIxESMZp72He/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1QWBUTzFigCUAWMblOH_qag?pwd=rivs)]
#### Testing
~~~
python main.py --data CSD --save_image True --mode test --data_dir your_path/CSD --test_model path_to_CSD_model

python main.py --data SRRS --save_image True --mode test --data_dir your_path/SRRS --test_model path_to_SRRS_model

python main.py --data Snow100K --save_image True --mode test --data_dir your_path/Snow100K --test_model path_to_Snow100K_model
~~~

