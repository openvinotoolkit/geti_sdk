import json
import os
from collections import Counter
from json import JSONDecodeError

from sc_api_tools import SCRESTClient
from sc_api_tools.nous.nous2sc import migrate_nous_chain
from sc_api_tools.rest_clients import ProjectClient

if __name__ == '__main__':
    ann_dir = "/home/lhogeweg/Documents/Datasets/diopsis21clean/annotation"
    labels = []
    for fn in os.listdir(ann_dir):
        try:
            j_ann = json.load(open(os.path.join(ann_dir, fn)))
            for j_shape in j_ann["data"]:
                labels.extend([x["name"] for x in j_shape["labels"]])
        except JSONDecodeError:
            print(fn)

    counts = Counter(labels).most_common()
    print(counts)
    sufficient_labels = set([x[0] for x in counts if x[1] >= 6])
    # exit(0)
    print(sufficient_labels)

    update_annotations = False
    if update_annotations:
        for fn in os.listdir(ann_dir):
            p = os.path.join(ann_dir, fn)
            j_ann = json.load(open(p))
            for j_shape in j_ann["data"]:
                j_shape["labels"] = [x for x in j_shape["labels"] if x["name"] in sufficient_labels]
            json.dump(j_ann, open(p, "w"), indent=2)

    j_project = json.load(open("/mnt/big/diopsis_s/project.json"))
    id_to_label = {}
    for j_label in j_project["pipeline"]["tasks"][3]["properties"][0]["user_value"]:
        print(j_label)
        print(j_label["name"], j_label["parent"], j_label["id"])
        id_to_label[j_label["id"]] = j_label

    labels_migrate = []

    # print(id_to_label)

    for id_, label in id_to_label.items():
        if label["name"] in sufficient_labels and label["name"] != "Object":
            migrate_label = {
                "name": label["name"],
            }
            if label["parent"] in id_to_label and label["name"] != "Empty Classification":
                migrate_label["parent_id"] = id_to_label[label["parent"]]["name"]
                migrate_label["group"] = "g_" + id_to_label[label["parent"]]["name"]
                print("migrate_label['group']", migrate_label["group"])
            labels_migrate.append(migrate_label)

    print(labels_migrate)

    # client = SCRESTClient(
    #     host="https://10.91.120.185",
    #     username="laurens.hogeweg@intel.com",
    #     password="@SCvision+LH"
    # )

    host = "https://vm40.openvino.ai"
    client = SCRESTClient(
        host=host,
        username="laurens.hogeweg@intel.com",
        password="@SCvision+LH",
        proxies={"https": "http://proxy-dmz.intel.com:912"}
    )

    # project_manager = ProjectClient(
    #     session=client.session, workspace_id=client.workspace_id
    # )
    # project = project_manager.get_or_create_project(
    #     project_name="diopsis",
    #     project_type='_to_'.join(['detection', 'classification']),
    #     labels=[['Object'], labels_migrate]
    # )
    # print(project)
    # with open("diopsis.json", "w") as f:
    #     json.dump(
    #         project_manager.session.get_rest_response(
    #             url=f"{project_manager.base_url}projects/{project.id}",
    #             method="GET",
    #         ), f, indent=2)
    # exit(0)

    # migrate_nous_project(
    #     rest_client=client,
    #     export_path='/mnt/big/backups/stickytraps_24nov',
    #     project_name='sticky traps detection',
    #     project_type='detection'
    # )

    migrate_nous_chain(
        rest_client=client,
        export_path='/home/lhogeweg/Documents/Datasets/diopsis21.zip',
        task_types=['detection', 'classification'],
        labels_per_task=[['Object'], labels_migrate],
        project_name='diopsis',
        temp_dir="/home/lhogeweg/Documents/Datasets/diopsis21clean"
    )
