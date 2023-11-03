# Linux x86_64 CPU or CUDA 12
Don't forget to copy the archived app into a folder called local-source.
When developing, install using something like:
```
sudo snap install --devmode mnist-vae_*_amd64.snap
```
When installing for production use:
```
snap install --dangerous mnist-vae_*_amd64.snap
snap connect mnist-vae:password-manager-service
snap connect mnist-vae:hardware-observe
```
Note that `--dangerous` is used because the snap has not been signed.
Uninstall using:
```
sudo snap remove --purge mnist-vae
```
Resources:
* https://snapcraft.io/docs/desktop-menu-support#:~:text=Desktop%20entry%20files%20in%20the%20%60snap%2Fgui%60%20directory&text=The%20desktop%20file%20and%20icon,name%3A%20entry%20in%20the%20snapcraft.
* https://snapcraft.io/docs/dump-plugin?_ga=2.80243987.502937871.1698610500-1953973932.1646255773
* https://ubuntu.com/tutorials/create-your-first-snap
