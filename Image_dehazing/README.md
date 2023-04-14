### Download the Datasets
- reside-indoor [[gdrive](https://drive.google.com/drive/folders/1pbtfTp29j7Ip-mRzDpMpyopCfXd-ZJhC?usp=sharing), [百度网盘](链接：https://pan.baidu.com/s/1jD-TU0wdtSoEb4ki-Cut2A?pwd=1lr0)(1lr0)]
- reside-outdoor [[gdrive](https://drive.google.com/drive/folders/1eL4Qs-WNj7PzsKwDRsgUEzmysdjkRs22?usp=sharing)]
- (Separate SOTS test set if needed) [[gdrive](https://drive.google.com/file/d/16j2dwVIa9q_0RtpIXMzhu-7Q6dwz_D1N/view?usp=sharing), [百度网盘](链接：https://pan.baidu.com/s/1R6qWri7sG1hC_Ifj-H6DOQ?pwd=o5sk)(o5sk)]
### Train on RESIDE-Indoor

~~~
python main.py --data Indoor --mode train --data_dir your_path/reside-indoor
~~~


### Train on RESIDE-Outdoor
~~~
python main.py --data Outdoor --mode train --data_dir your_path/reside-outdoor --batch_size 8 --num_epoch 30  --save_freq 1 --valid__freq 1
~~~


### Evaluation
#### Download the model 
- SOTS-Indoor [[gdrive](https://drive.google.com/file/d/1UsNrGkWie-PKXcGSA6oFkt0WgnW8Bqsi/view?usp=sharing), [百度网盘](链接：https://pan.baidu.com/s/1Z3La73rya9GVQR4STYk_XA?pwd=ods2)(ods2))]
- SOTS-Outdoor [[gdrive](https://drive.google.com/file/d/16lbhL3fqHeVu-aPkmFSUnaHWQKxmhPz6/view?usp=sharing), [百度网盘](链接：https://pan.baidu.com/s/1NEcAus7lOuvtot-00sjxBg?pwd=hkab)(hkab)]
#### Testing on SOTS-Indoor
~~~
python main.py --data Indoor --save_image True --mode test --data_dir your_path/reside-indoor --test_model path_to_its_model
~~~
#### Testing on SOTS-Outdoor
~~~
python main.py --data Outdoor --save_image True --mode test --data_dir your_path/reside-outdoor --test_model path_to_ots_model
~~~