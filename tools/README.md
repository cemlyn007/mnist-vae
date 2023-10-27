# Notes:
If after building the application on Mac and running it you see a generic icon on the left side bar when the app is minimised,
it could be an issue with Mac's icon cache. To fix this, run the following command in the terminal:
```
sudo rm -rfv /Library/Caches/com.apple.iconservices.store
```
