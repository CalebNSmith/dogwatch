import os
import sys
import sqlite3
import subprocess

def get_row(db_file, image_id):
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM prediction WHERE image_id=?", (image_id,))
    row = cur.fetchone()
    cur.execute("SELECT count(*) FROM prediction WHERE human_label IS NULL")
    left = cur.fetchone()[0]
    cur.close()
    conn.close()
    return row, left

def pretty_row(row, left):
    max_percent = 0
    place = None
    for n in range(1, 4):
        if row[n] > max_percent:
            max_percent = row[n]
            place = n
    if place == 1:
        predicted = 'laying'
    elif place == 2:
        predicted = 'sitting'
    else:
        predicted = 'standing'
    
    if row[4] is None:
        label = 'unlabeled'
    else:
        label = row[4]
    print(predicted, max_percent, label, row[6])
    print('%d left to be labeled' % (left,))

def update_label(db_file, image_id, label):
    conn = sqlite3.connect(db_file)
    conn.cursor().execute("UPDATE prediction SET human_label=? WHERE image_id=?", (label, image_id,))
    conn.commit()
    conn.close()
    
def update_dataset(db_file, image_id, dataset):
    conn = sqlite3.connect(db_file)
    conn.cursor.execute("UPDATE prediction SET human_dataset=? WHERE image_id=?", (dataset, image_id))
    conn.commit()
    conn.close()

def main(db_file, image_id, action=None):
    image_id = image_id.strip('.jpeg')
    if action is None:
        row, left = get_row(db_file, image_id)
        pretty_row(row, left)
    if action in ['laying', 'sitting', 'standing', 'other']:
        update_label(db_file, image_id, action)
    if action in ['dev', 'test', 'eyeball']:
        update_dataset(db_file, image_id, action)

if __name__ == '__main__':
    subprocess.call('clear')
    if len(sys.argv) < 4:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
