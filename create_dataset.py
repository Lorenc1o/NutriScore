import requests
import json

base_url = 'https://world.openfoodfacts.org/api/v0/'
endpoint = 'search'
params = {
    'page_size': 200,
    'countries_tags_en': 'united-kingdom',
    'categories_tags': 'plant-based-foods',
    'fields': 'id,product_name,nutriscore_grade,nutriments,fruits-vegetables-nuts_100g',

}

response = requests.get(base_url + endpoint, params=params)

if response.status_code == 200:
    data = response.json()
    products = data['products']

    print('Number of products:', len(products))
    
    with open('data.csv', 'w') as f:
        f.write('id;name;nutriscore;energy;fat_g;sugar_g;sodium_mg;perc_fruit;fibers_g;proteins_g\n')
        n_zeros = 0
        for product in products:
            try:
                id = product['id']
                name = product['product_name']
                nutriscore = product['nutriscore_grade']
                energy = product['nutriments']['energy-kcal_100g']
                # kcal to kJ
                energy = energy * 4.184

                if 'fat_100g' in product['nutriments']:
                    fat_g = product['nutriments']['fat_100g']
                else:
                    fat_g = 0
                    n_zeros += 1
                
                if 'sugars_100g' in product['nutriments']:
                    sugar_g = product['nutriments']['sugars_100g']
                else:
                    sugar_g = 0
                    n_zeros += 1

                if 'sodium_100g' in product['nutriments']:
                    sodium_mg = product['nutriments']['sodium_100g']*1000
                else:
                    sodium_mg = 0
                    n_zeros += 1

                try:
                    perc_fruit = product['nutriments']['fruits-vegetables-nuts-estimate_100g']
                except KeyError:
                    try:
                        perc_fruit = product['fruits-vegetables-nuts_100g']
                    except KeyError:
                        perc_fruit = 0
                        n_zeros += 1
                
                if 'fiber_100g' in product['nutriments']:
                    fibers_g = product['nutriments']['fiber_100g']
                else:
                    fibers_g = 0
                    n_zeros += 1

                if 'proteins_100g' in product['nutriments']:
                    proteins_g = product['nutriments']['proteins_100g']
                else:
                    proteins_g = 0
                    n_zeros += 1

                f.write(f'{id};{name};{nutriscore};{energy};{fat_g};{sugar_g};{sodium_mg};{perc_fruit};{fibers_g};{proteins_g}\n')
            except KeyError as e:
                print('KeyError:', e)
                pass
        print('Number of zeros:', n_zeros)

else:
    print('Failed to get data:', response.status_code)