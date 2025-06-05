#!/bin/bash

max_retries=3
attempt=0
success=false

mkdir -p ./output

while [ $attempt -lt $max_retries ] && [ "$success" = false ]; do
  attempt=$((attempt + 1))
  echo "Attempt $attempt of $max_retries" >> ./output/hash_check_errors.log
  
  (timeout 120 find /home -type f -exec sha256sum {} \; > hashes.txt 2>> ./output/hash_check_errors.log && \
   diff hashes.txt trusted_hashes.txt > results.txt 2>> ./output/hash_check_errors.log)
  
  if [ $? -eq 0 ]; then
    success=true
  else
    echo "Attempt $attempt failed" >> ./output/hash_check_errors.log
    [ $attempt -lt $max_retries ] && sleep 5
  fi
done

if [ "$success" = false ]; then
  echo "All attempts failed" >> ./output/hash_check_errors.log
  exit 1
fi 