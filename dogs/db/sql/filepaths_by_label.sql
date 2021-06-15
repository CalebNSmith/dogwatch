WITH label_enum 
  AS (SELECT label.id
        FROM label
       WHERE label.name = ?) 
SELECT filename, (SELECT * FROM label_enum)
  FROM image
 WHERE image.id IN
       (SELECT image_id
          FROM image_context
         WHERE label_id = (SELECT * FROM label_enum)
           AND training_set = ?)
 ORDER BY RANDOM();
