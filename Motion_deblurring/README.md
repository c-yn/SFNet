### Download the Datasets
- Gopro [[gdrive](https://drive.google.com/file/d/1y_wQ5G5B65HS_mdIjxKYTcnRys_AGh5v/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1eNCvqewdUp15-0dD2MfJbg?pwd=ea0r)]
- HIDE [[gdrive](https://drive.google.com/file/d/13CoUG0YktPGzVagOipoo43NMZclOG7J2/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1F70f040UWAaeofSie_zMow?pwd=c8lv)]
### Training on GoPro 
~~~
python main.py  --data_dir your_path/GOPRO
~~~
### Evaluation
#### Download the model
-  [[gdrive](https://drive.google.com/drive/folders/1OJv9d6e90hlpDSyo8oJY-END3xj4nUmg?usp=sharing)(GoPro/RSBlur), [Baidu](https://pan.baidu.com/s/1PXQgpI-h-Epiaiy9wy3CUg?pwd=10ne)(GoPro)]
-  
#### Testing on GoPro
~~~
python main.py --data GOPRO --mode test --data_dir your_path/GOPRO --test_model path_to_gopro_model --save_image True
~~~
#### Testing on HIDE
~~~
python main.py --data HIDE --mode test --data_dir your_path/HIDE --test_model path_to_gopro_model --save_image True
~~~
#### Testing on RSBlur
~~~
python main.py --data RSBlur --mode test --data_dir your_path/RSBlur --test_model path_to_RSBlur_model --save_image False
~~~
Pleae set 'save_image' as Ture to get the resulting images, and then using the official evaluation code of RSBlur to produce PSNR/SSIM.
