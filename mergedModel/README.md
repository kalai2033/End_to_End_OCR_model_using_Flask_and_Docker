# MyOCR - CRAFT text detection + Text recognition
Download the pretrained text detection model (craft_mlt_25k.pth) from https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view?usp=drive_open.

Download the pretrained text recognition model (TPS-ResNet-BiLSTM-Attn.pth) from https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW

Place both the models inside the mergedModel folder

To extract text from the input image run !python extracttext.py --path (input image)