#!/bin/bash

cd deploy
rm -r out

echo "Creating deployment package..."
mkdir out
mkdir -p ./out/backend
mkdir -p ./out/backend/api
mkdir -p ./out/nginx
mkdir -p ./out/logs

cp ./docker-compose.yml ./out/
cp -r ./backend/ ./out/backend/
cp -r ./nginx/ ./out/nginx/

echo $'\nUpdating resource URIs for production...'
sed -i '' 's/http:\/\/localhost:5000/https:\/\/detect.alexpotter.net\/detector/g' ../src/api/APIClient.tsx
sed -i '' 's/..\/..\/save_temp.h5/.\/save_temp.h5/g' ../api/api.py
sed -i '' 's/..\/..\/train_encoder/.\/train_encoder/g' ../api/api.py

cp -r ../api/ ./out/backend/api/

rm ../api/parse_training_input.py
rm ../api/util.py
rm ../api/labels.txt

cp ../../parse_training_input.py ./out/backend/api/
cp ../../util.py ./out/backend/api/
cp ../../labels.txt ./out/backend/api/
cp ../../save_temp.h5 ./out/backend/api/
cp ../../train_encoder.subwords ./out/backend/api/

cd ..
npm run-script build
cp -r ./build ./deploy/out/nginx/web/

cd deploy
zip -r vulndetect_production.zip ./out/*

echo $'\nReverting resource URIs for development...'
sed -i '' 's/https:\/\/detect.alexpotter.net\/detector/http:\/\/localhost:5000/g' ../src/api/APIClient.tsx
sed -i '' 's/.\/save_temp.h5/..\/..\/save_temp.h5/g' ../api/api.py
sed -i '' 's/.\/train_encoder/..\/..\/train_encoder/g' ../api/api.py

echo $'\n\nSaved deployment zip to ./deploy/vulndetect_production.zip'
echo "Transfer to the server and start with 'docker-compose up --build -d'"