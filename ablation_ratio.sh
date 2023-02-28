
python tools/benchmark.py --model-name DeeplabV3Plus --backbone-name MobileNetV3-large
python tools/benchmark.py --model-name SOSNet --backbone-name MobileNetV3-large
python tools/benchmark.py --model-name SegFormer --backbone-name MiT-B0


python tools/train_sosnet.py --cfg configs/UperNet/vaihingen_mbv3l.yaml --hier false --soem true --ratio 0.0
python tools/train_sosnet.py --cfg configs/UperNet/vaihingen_mbv3l.yaml --hier false --soem true --ratio 0.1
python tools/train_sosnet.py --cfg configs/UperNet/vaihingen_mbv3l.yaml --hier false --soem true --ratio 0.2
python tools/train_sosnet.py --cfg configs/UperNet/vaihingen_mbv3l.yaml --hier false --soem true --ratio 0.3
python tools/train_sosnet.py --cfg configs/UperNet/vaihingen_mbv3l.yaml --hier false --soem true --ratio 0.4
python tools/train_sosnet.py --cfg configs/UperNet/vaihingen_mbv3l.yaml --hier false --soem true --ratio 0.5
python tools/train_sosnet.py --cfg configs/UperNet/vaihingen_mbv3l.yaml --hier false --soem true --ratio 0.6
python tools/train_sosnet.py --cfg configs/UperNet/vaihingen_mbv3l.yaml --hier false --soem true --ratio 0.7
python tools/train_sosnet.py --cfg configs/UperNet/vaihingen_mbv3l.yaml --hier false --soem true --ratio 0.8
python tools/train_sosnet.py --cfg configs/UperNet/vaihingen_mbv3l.yaml --hier false --soem true --ratio 0.9
python tools/train_sosnet.py --cfg configs/UperNet/vaihingen_mbv3l.yaml --hier false --soem true --ratio 1.0
