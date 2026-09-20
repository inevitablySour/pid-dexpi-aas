import base64, zlib, urllib.parse, re, sys, os

def decode_drawio(path):
    with open(path, encoding='utf-8') as f:
        content = f.read()
    diagrams = re.findall(r'<diagram[^>]*>(.*?)</diagram>', content, re.S)
    out = []
    for payload in diagrams:
        payload = payload.strip()
        if not payload:
            out.append('')
            continue
        try:
            raw = base64.b64decode(payload)
            inflated = zlib.decompress(raw, -15)
            xml = urllib.parse.unquote(inflated.decode('utf-8'))
        except Exception as e:
            # maybe already plain xml (uncompressed)
            xml = payload
        out.append(xml)
    return out

if __name__ == '__main__':
    src = sys.argv[1]
    dst_dir = sys.argv[2]
    os.makedirs(dst_dir, exist_ok=True)
    diagrams = decode_drawio(src)
    base = os.path.splitext(os.path.basename(src))[0]
    for i, xml in enumerate(diagrams):
        outpath = os.path.join(dst_dir, f"{base}__diagram{i}.xml")
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write(xml)
        print(outpath, len(xml))
