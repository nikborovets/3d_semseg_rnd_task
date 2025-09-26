1. Install `jq`:

```bash
sudo apt-get update
sudo apt-get install -y jq
```

2. Get the direct download link:

```bash
curl -s "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.360.yandex.ru/d/-oVuujK-B4LryQ" | jq -r '.href'
```

3. Download the archive:

```bash
wget -O folder.zip "PASTE_LINK_FROM_PREVIOUS_COMMAND"
```

4. Extract:

```bash
unzip folder.zip -d folder
```
