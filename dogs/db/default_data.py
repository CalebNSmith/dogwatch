import itertools
import json
import re
import os

# DOG
def ids_with_images():
    img_dir = '/data/old/dog/raw/'
    ids = set(os.listdir(img_dir))
    rc_dir = '/home/dan/allpaws/report_cards/'
    report_card_files = [
        f"{rc_dir}/{rc_file}" 
        for rc_file in os.listdir(rc_dir) 
        if rc_file.split('-all')[0] in ids
    ] 
    return report_card_files

def dog_from_report_card(report_card):
    regex_form_data = re.compile("var formData")
    regex_animal = re.compile("var animal ")
    dog = {}
    with open(report_card, 'r') as f:
        for line in f:
            form_data = regex_form_data.search(line)
            animal = regex_animal.search(line)
            if form_data is not None:
                x = line.split('=')[1]
                x = x.split('do_you_up_keep_your_dog_s_flea_and_tick_medicine')[0]
                x = x.strip(' ","')
                x = x + '"}'
                x = json.loads(x)
                try:
                    dog['colors'] = x['color_and_markings_gingr']
                except:
                    dog['colors'] = None
            if animal is not None:
                x = line.split('=')[1]
                x = x.strip()
                x = x.strip(';')
                x = json.loads(x)
                dog['id'] = x['a_id']
                dog['breed'] = x['breed_name']
                dog['name'] = x['animal_name']
                dog['weight'] = x['weight']
    if dog == {}:
        return None
    return dog

dogs = [dog_from_report_card(rc_file) for rc_file in ids_with_images() if not None]

# Breed
breeds = [
    ('Affenpinschers',),
    ('Afghan Hounds',),
    ('Airedale Terriers',),
    ('Akitas',),
    ('Alaskan Malamutes',),
    ('American English Coonhounds',),
    ('American Eskimo Dogs',),
    ('American Foxhounds',),
    ('American Hairless Terriers',),
    ('American Staffordshire Terriers',),
    ('Anatolian Shepherd Dogs',),
    ('Australian Cattle Dogs',),
    ('Australian Shepherds',),
    ('Australian Terriers',),
    ('Azawakhs',),
    ('Basenjis',),
    ('Basset Hounds',),
    ('Beagles',),
    ('Bearded Collies',),
    ('Beaucerons',),
    ('Bedlington Terriers',),
    ('Belgian Malinois',),
    ('Belgian Sheepdogs',),
    ('Belgian Tervuren',),
    ('Bergamasco Sheepdogs',),
    ('Berger Picards',),
    ('Bernese Mountain Dogs',),
    ('Bichons Frises',),
    ('Black Russian Terriers',),
    ('Black and Tan Coonhounds',),
    ('Bloodhounds',),
    ('Bluetick Coonhounds',),
    ('Boerboels',),
    ('Border Collies',),
    ('Border Terriers',),
    ('Borzois',),
    ('Boston Terriers',),
    ('Bouviers des Flandres',),
    ('Boxers',),
    ('Briards',),
    ('Brittanys',),
    ('Brussels Griffons',),
    ('Bull Terriers',),
    ('Bulldogs',),
    ('Bullmastiffs',),
    ('Cairn Terriers',),
    ('Canaan Dogs',),
    ('Cane Corso',),
    ('Cardigan Welsh Corgis',),
    ('Cavalier King Charles Spaniels',),
    ('Cesky Terriers',),
    ('Chihuahuas',),
    ('Chinese Crested',),
    ('Chinese Shar-Pei',),
    ('Chinooks',),
    ('Chow Chows',),
    ('Cirnechi dell’Etna',),
    ('Collies',),
    ('Coton de Tulear',),
    ('Dachshunds',),
    ('Dalmatians',),
    ('Dandie Dinmont Terriers',),
    ('Doberman Pinschers',),
    ('Dogues de Bordeaux',),
    ('English Foxhounds',),
    ('English Toy Spaniels',),
    ('Entlebucher Mountain Dogs',),
    ('Finnish Lapphunds',),
    ('Finnish Spitz',),
    ('Fox Terriers (Smooth)',),
    ('Fox Terriers (Wire)',),
    ('French Bulldogs',),
    ('German Pinschers',),
    ('German Shepherd Dogs',),
    ('Giant Schnauzers',),
    ('Glen of Imaal Terriers',),
    ('Grand Basset Griffon Vendeens',),
    ('Great Danes',),
    ('Great Pyrenees',),
    ('Greater Swiss Mountain Dogs',),
    ('Greyhounds',),
    ('Harriers',),
    ('Havanese',),
    ('Ibizan Hounds',),
    ('Icelandic Sheepdogs',),
    ('Irish Terriers',),
    ('Irish Wolfhounds',),
    ('Italian Greyhounds',),
    ('Japanese Chin',),
    ('Keeshonden',),
    ('Kerry Blue Terriers',),
    ('Komondorok',),
    ('Kuvaszok',),
    ('Lagotti Romagnoli',),
    ('Lakeland Terriers',),
    ('Leonbergers',),
    ('Lhasa Apsos',),
    ('Lowchen',),
    ('Maltese',),
    ('Manchester Terriers',),
    ('Mastiffs',),
    ('Miniature American Shepherds',),
    ('Miniature Bull Terriers',),
    ('Miniature Pinschers',),
    ('Miniature Schnauzers',),
    ('Mix',),
    ('Neapolitan Mastiffs',),
    ('Nederlandse Kooikerhondjes',),
    ('Newfoundlands',),
    ('No Breed',),
    ('Norfolk Terriers',),
    ('Norwegian Buhunds',),
    ('Norwegian Elkhounds',),
    ('Norwegian Lundehunds',),
    ('Norwich Terriers',),
    ('Old English Sheepdogs',),
    ('Otterhounds',),
    ('Papillons',),
    ('Parson Russell Terriers',),
    ('Pekingese',),
    ('Pembroke Welsh Corgis',),
    ('Petits Bassets Griffons Vendeens',),
    ('Pharaoh Hounds',),
    ('Pit Bulls'),
    ('Plott Hounds',),
    ('Pointers (German Shorthaired)',),
    ('Pointers (German Wirehaired)',),
    ('Pointers',),
    ('Polish Lowland Sheepdogs',),
    ('Pomeranians',),
    ('Poodles',),
    ('Portuguese Podengo Pequenos',),
    ('Portuguese Water Dogs',),
    ('Pugs',),
    ('Pulik',),
    ('Pumik',),
    ('Pyrenean Shepherds',),
    ('Rat Terriers',),
    ('Redbone Coonhounds',),
    ('Retrievers (Chesapeake Bay)',),
    ('Retrievers (Curly-Coated)',),
    ('Retrievers (Flat-Coated)',),
    ('Retrievers (Golden)',),
    ('Retrievers (Labrador)',),
    ('Retrievers (Nova Scotia Duck Tolling)',),
    ('Rhodesian Ridgebacks',),
    ('Rottweilers',),
    ('Russell Terriers',),
    ('Salukis',),
    ('Samoyeds',),
    ('Schipperkes',),
    ('Scottish Deerhounds',),
    ('Scottish Terriers',),
    ('Sealyham Terriers',),
    ('Setters (English)',),
    ('Setters (Gordon)',),
    ('Setters (Irish Red and White)',),
    ('Setters (Irish)',),
    ('Shetland Sheepdogs',),
    ('Shiba Inu',),
    ('Shih Tzu',),
    ('Siberian Huskies',),
    ('Silky Terriers',),
    ('Skye Terriers',),
    ('Sloughis',),
    ('Soft Coated Wheaten Terriers',),
    ('Spaniels (American Water)',),
    ('Spaniels (Boykin)',),
    ('Spaniels (Clumber)',),
    ('Spaniels (Cocker)',),
    ('Spaniels (English Cocker)',),
    ('Spaniels (English Springer)',),
    ('Spaniels (Field)',),
    ('Spaniels (Irish Water)',),
    ('Spaniels (Sussex)',),
    ('Spaniels (Welsh Springer)',),
    ('Spanish Water Dogs',),
    ('Spinoni Italiani',),
    ('St. Bernards',),
    ('Staffordshire Bull Terriers',),
    ('Standard Schnauzers',),
    ('Swedish Vallhunds',),
    ('Tibetan Mastiffs',),
    ('Tibetan Spaniels',),
    ('Tibetan Terriers',),
    ('Toy Fox Terriers',),
    ('Treeing Walker Coonhounds',),
    ('Vizslas',),
    ('Weimaraners',),
    ('Welsh Terriers',),
    ('West Highland White Terriers',),
    ('Whippets',),
    ('Wirehaired Pointing Griffons',),
    ('Wirehaired Vizslas',),
    ('Xoloitzcuintli',),
    ('Yorkshire Terriers',)
]
