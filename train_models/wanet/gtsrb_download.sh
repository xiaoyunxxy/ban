wget -P . https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
wget -P . https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
wget -P . https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip 
mkdir ./GTSRB;
mkdir ./GTSRB/Train;
mkdir ./GTSRB/Test;
mkdir ./temps;
unzip ./GTSRB_Final_Training_Images.zip -d ./temps/Train;
unzip ./GTSRB_Final_Test_Images.zip -d ./temps/Test;
mv ./temps/Train/GTSRB/Final_Training/Images/* ./GTSRB/Train;
mv ./temps/Test/GTSRB/Final_Test/Images/* ./GTSRB/Test;
unzip ./GTSRB_Final_Test_GT.zip -d ./GTSRB/Test/;
rm -r ./temps;
rm ./*.zip;
echo "Download Completed";
