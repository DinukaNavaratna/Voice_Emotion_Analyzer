## About

This is a simple web application that can be used to upload an audio file containing vocals and detect the emotion of the vocals. The uploaded audio file will be chopped into mini clips and analyze each mini clip for the emotion. The accuracy is not very good as this was done within a limited time period and with a limited dataset provided.<br>
The project was completed according to the requirements of a friend of mine. Frontend of the application was provided and I assume it is a template based design taken from somewhere.<br>
Try using .wav audio files when testing because it might give an error if you try to process other types of files due to a ffmpeg issue. There are many more improvements that can be done in this project and is not guranteed to be 100% error free.


## Run the Application

1. Create a Virtual Environement (Optional)
2. Delete the .gitignore file inside the src/SER/Audio_Speech_Actors_01-24
3. Delete the .gitignore file inside the src/SER/RAV
4. Download the RAR file using https://drive.google.com/file/d/1l3VQngVHnoKAWDopD34uFszILNt74yCJ/view?usp=sharing
5. Download the RAR file using https://drive.google.com/file/d/1GX5k0IF5PXmBFILPw0L-7IlDLAlqrW71/view?usp=sharing
6. Extract the content inside the root directory of Audio_Speech_Actors_01-24.rar into src/SER/Audio_Speech_Actors_01-24
7. Extract the content inside the root directory of RAV.rar into src/SER/RAV
8. Install the requirements inside the requirements.in & requirements.txt using PIP (Ex: pip install -r requirements.txt)
9. Run the app.py (Ex: Python app.py)
10. The webpage can be loaded using the emotion-detect.html file inside src/web/emotion-detection/


## Snapshots
![Dinuka Navaratna - Voice Emotion Analyzer (Snapshot 1)](https://user-images.githubusercontent.com/26020039/162551058-d94103d6-59e5-45cd-8b8d-d138d1aa6e03.png)

![Dinuka Navaratna - Voice Emotion Analyzer (Snapshot 2)](https://user-images.githubusercontent.com/26020039/162551064-788b2e77-2f59-454f-abf7-cf2a12087e45.png)

![Dinuka Navaratna - Voice Emotion Analyzer (Snapshot 3)](https://user-images.githubusercontent.com/26020039/162551067-e1b88105-8b1b-43bb-9f4a-b51504a1607c.png)
