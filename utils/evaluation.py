import os
import os.path as osp


def export_results(results, output_dir, name):
    lines = []
    frameIds = sorted(list(results.keys()))
    for frameId in frameIds:
        tracks = results[frameId]
        if len(tracks) == 0:
            continue
        for track in tracks:
            tid = track['id']
            box = track['box']
            xmin = box[0]
            ymin = box[1]
            width = box[2]-box[0]
            height = box[3]-box[1]
            line = f"{frameId},{tid},{xmin},{ymin},{width},{height},-1,-1,-1,-1"
            lines.append(line)

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    output_path = osp.join(output_dir, f'{name}.txt')
    with open(output_path, 'w') as f:
        content = '\n'.join(lines)
        f.write(content)

    print(f"Save tracking result to '{output_path}'")
