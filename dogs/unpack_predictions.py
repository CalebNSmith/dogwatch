import sys

from db.dbwrapper import Database
from db import DojoImage

if len(sys.argv) < 2:
    print("FAIL")
    exit()

# Check if images in checked_out, Fail if not
DojoImage().un_check_out_images(sys.argv[1])

# Put images into main database
with Database(sys.argv[1] + '.db') as db:
    laying = db.query("SELECT image_id FROM prediction WHERE human_label='laying'")
    sitting = db.query("SELECT image_id FROM prediction WHERE human_label='sitting'")
    standing = db.query("SELECT image_id FROM prediction WHERE human_label='standing'")
    other = db.query("SELECT image_id FROM prediction WHERE human_label='other'")

print(len(laying), len(sitting), len(standing), len(other))
with Database() as db:
    db.query("UPDATE dojo_image SET label=0 WHERE image_id=?", [(l['image_id'],) for l in laying])
    db.query("UPDATE dojo_image SET label=1 WHERE image_id=?", [(s['image_id'],) for s in sitting])
    db.query("UPDATE dojo_image SET label=2 WHERE image_id=?", [(s['image_id'],) for s in standing])
    # Set other apart so don't predict on them again next time
    db.query("UPDATE dojo_image SET label=69 WHERE image_id=?", [(o['image_id'],) for o in other])
