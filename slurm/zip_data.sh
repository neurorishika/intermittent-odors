#!/bin/sh
zip -r "zip_$1.zip" Data/
mv "zip_$1.zip" "../data/zip_$1.zip"