import pandas as pd
import streamlit as st
import joblib
import folium


# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(layout="wide")
st.title("Previsão de preço Airbnb - Rio de Janeiro")

# --- CRIAÇÃO DAS ABAS ---
tab_previsao, = st.tabs(["Ferramenta de Previsão"])

# --- ABA 1: FERRAMENTA DE PREVISÃO ---
with tab_previsao:

    st.header("Preveja o Valor de um Imóvel")

    mapa_nomes = {
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'accommodates': 'Quantos hóspedes o imóvel pode acomodar',
        'bathrooms': 'Nº de Banheiros (lavabo equivale a 0.5)',
        'bedrooms': 'Nº de Quartos',
        'beds': 'Nº de Camas',
        'ano': 'Ano',
        'mes': 'Mês',
        'numero_de_amenities': 'Nº de Comodidades (Exemplo: TV, Internet, Ar condicionado,...)',
        'host_is_superhost': 'Anfitrião é Superhost', 
        'instant_bookable': 'Reserva Instantânea',
        'property_type': 'Tipo de Propriedade',
        'room_type': 'Tipo de Quarto',
        'cancellation_policy': 'Política de Cancelamento'
    }

    x_listas = {'property_type': ['Apartment',
                                  'Bed and breakfast',
                                  'Condominium',
                                  'Guest suite',
                                  'Hostel',
                                  'House',
                                  'Serviced apartment',
                                  'Loft',
                                  'Outros'],
                'cancellation_policy': ['flexible',
                                        'moderate',
                                        'strict',
                                        'strict_14_with_grace_period']
    }

    mapa_traducoes = {'property_type': {'Apartment': 'Apartamento',
                                        'Bed and breakfast': 'Cama e Café (B&B)',
                                        'Condominium': 'Condomínio',
                                        'Guest suite': 'Suíte de Hóspedes',
                                        'Hostel': 'Hostel',
                                        'House': 'Casa',
                                        'Serviced apartment': 'Apart-hotel',
                                        'Loft': 'Loft',
                                        'Outros': 'Outros'},
                      'cancellation_policy': {'flexible': 'Flexível',
                                              'moderate': 'Moderada',
                                              'strict': 'Rigorosa',
                                              'strict_14_with_grace_period': 'Rigorosa (14 dias com carência)'
                            }}


    dicionario_features = {}
    for item in x_listas:
        for valor in x_listas[item]:
            dicionario_features[f'{item}_{valor}'] = 0



    # Criamos 2 colunas: a primeira terá 1/4 da largura e a segunda 3/4
    col_ano, col_vazia = st.columns([1, 3])

    # Colocamos o selectbox dentro da primeira coluna (a menor)
    with col_ano:
        ano_selecionado = st.selectbox(
            label=mapa_nomes['ano'], 
            options=[2018, 2019, 2020]
    )


   # ano_selecionado = st.selectbox(label=mapa_nomes['ano'], options=[2018, 2019, 2020])

 # ----------------------------------------------------------------------------------
    x_numericos = {'latitude': 0.0,
                   'longitude': 0.0,
                   'accommodates': 1,
                   'bathrooms': 1.0,
                   'bedrooms': 1,
                   'beds': 1,
                   'mes': 1,
                   'numero_de_amenities': 1}
    
    mapa_restricoes = {
        'host_listings_count': (0, 6),
        'accommodates': (1, 9),
        'bathrooms': (1.0, 4.0),
        'bedrooms': (0, 3),
        'beds': (0, 6),
        'numero_de_amenities': (1, 30),
        'mes': (1, 12),
        'room_type':(0,2)
    }

    # Pega a lista de campos que vamos usar
    campos_numericos = list(x_numericos.keys())

    # --- Grupo 1: Estrutura e Capacidade ---
    st.caption('Defina a capacidade e estrutura principal do imóvel')
    col1, col2, col3, col4 = st.columns(4)

    # Dicionário para mapear colunas a campos deste grupo
    mapa_estrutura = {
        col1: 'accommodates',
        col2: 'bedrooms',
        col3: 'beds',
        col4: 'bathrooms'
    }

    for col, item in mapa_estrutura.items():
        with col:
            # A lógica para criar o widget é a mesma, mas agora está mais organizada
            label_amigavel = mapa_nomes[item]
            min_val, max_val = mapa_restricoes.get(item, (None, None))
            valor_inicial = min_val if min_val is not None else 0.0

            if item == 'bathrooms':
                valor = st.number_input(label=label_amigavel, min_value=float(min_val), max_value=float(max_val), value=float(valor_inicial), step=0.5, key=f'pred_{item}')
            else:
                valor = st.number_input(label=label_amigavel, min_value=int(min_val), max_value=int(max_val), value=int(valor_inicial), step=1, key=f'pred_{item}')
            x_numericos[item] = valor


# ----------------------------------------------------------------------------------
    st.info("""
        **Limites da cidade:**
        * **Latitude:** entre -23.08332 e -22.74454
        * **Longitude:** entre -43.79544 e -43.09913
        """)

    # --- Grupo 2: Detalhes, Localização e Comodidades ---
    col_detalhes1, col_detalhes2 = st.columns(2)

    with col_detalhes1:
        st.caption('Coordenadas Geográficas ')

        # Latitude
        item_lat = 'latitude'
        label_lat = mapa_nomes[item_lat]
        x_numericos[item_lat] = st.number_input(label=label_lat, value=0.0, step=0.00001, format="%.5f", key=f'pred_{item_lat}')

        # Longitude
        item_lon = 'longitude'
        label_lon = mapa_nomes[item_lon]
        x_numericos[item_lon] = st.number_input(label=label_lon, value=0.0, step=0.00001, format="%.5f", key=f'pred_{item_lon}')


# ----------------------------------------------------------------------------------

    with col_detalhes2:
        st.caption('Data da Reserva e Comodidades')
        # Mês
        item_mes = 'mes'
        label_mes = mapa_nomes[item_mes]
        min_mes, max_mes = mapa_restricoes.get(item_mes, (1, 12))
        x_numericos[item_mes] = st.number_input(label=label_mes, min_value=min_mes, max_value=max_mes, value=1, step=1, key=f'pred_{item_mes}')

        # Número de Comodidades
        item_amenities = 'numero_de_amenities'
        label_amenities = mapa_nomes[item_amenities]
        min_amenities, max_amenities = mapa_restricoes.get(item_amenities, (0, 50))
        x_numericos[item_amenities] = st.number_input(label=label_amenities, min_value=min_amenities, max_value=max_amenities, value=1, step=1, key=f'pred_{item_amenities}')
        
# ----------------------------------------------------------------------------------

    x_tf = {'host_is_superhost': 0,
           'instant_bookable': 0}

    st.divider()
    col1_tf, col2_tf = st.columns(2)


    with col1_tf:
        item = 'host_is_superhost'
        label_amigavel = mapa_nomes[item]
        valor = st.selectbox(label=label_amigavel, options=('Não', 'Sim'), key=f'pred_{item}')
        x_tf[item] = 1 if valor == 'Sim' else 0


    with col2_tf:
        item = 'instant_bookable'
        label_amigavel = mapa_nomes[item]
        valor = st.selectbox(label=label_amigavel, options=('Não', 'Sim'), key=f'pred_{item}')
        x_tf[item] = 1 if valor == 'Sim' else 0

# ----------------------------------------------------------------------------------

    # --- PRIMEIRA LINHA: Tipo de Quarto e Política de Cancelamento ---
    col1_listas, col2_listas = st.columns(2)
    campos_listas = list(x_listas.keys())

    # Mapa para as opções do room_type
    mapa_opcoes_room_type = {
        'Entire home/apt': 2,
        'Private or Hotel Room': 1,
        'Shared room': 0
    }

    mapa_traducoes_room_type = {
    'Entire home/apt': 'Casa/Apto Inteiro',
    'Private or Hotel Room': 'Quarto Privado ou de Hotel',
    'Shared room': 'Quarto Compartilhado'
    }

    with col1_listas:
        label_rt = mapa_nomes['room_type']
        # Pega as opções ORIGINAIS (em inglês)
        opcoes_originais = list(mapa_opcoes_room_type.keys())

        # Cria uma lista com as opções TRADUZIDAS para mostrar ao usuário
        # A função .get(key, key) garante que se uma tradução não for encontrada, ele mostra o original
        opcoes_traduzidas = [mapa_traducoes_room_type.get(key, key) for key in opcoes_originais]

        selecao_usuario_pt  = st.selectbox(label=label_rt, options=opcoes_traduzidas, key='pred_room_type_unico')

        # Encontra a chave original (em inglês) correspondente à seleção em português
        #  Isso faz o "caminho de volta" da tradução para o valor que o código precisa
        chave_original_en = next(
                key for key, value in mapa_traducoes_room_type.items() if value == selecao_usuario_pt
            )

        # Usa a chave original (em inglês) para obter o valor numérico (0, 1, 2)
        valor_numerico = mapa_opcoes_room_type[chave_original_en]

        x_numericos['room_type'] = valor_numerico

    with col2_listas:
        # A POLÍTICA DE CANCELAMENTO CONTINUA AQUI
        item = 'cancellation_policy'
        if item in campos_listas:
            label_amigavel = mapa_nomes[item]
            opcoes_traduzidas = list(mapa_traducoes[item].values())
            selecao_pt = st.selectbox(label=label_amigavel, options=opcoes_traduzidas, key=f'pred_{item}')
            valor_original_en = next(key for key, value in mapa_traducoes[item].items() if value == selecao_pt)
            dicionario_features[f'{item}_{valor_original_en}'] = 1

# ----------------------------------------------------------

    col_prop, col_vazia = st.columns(2)

    with col_prop:
        # A LÓGICA DO TIPO DE PROPRIEDADE FOI MOVIDA PARA CÁ
        item = 'property_type'
        if item in campos_listas:
            label_amigavel = mapa_nomes[item]
            opcoes_traduzidas = list(mapa_traducoes[item].values())
            selecao_pt = st.selectbox(label=label_amigavel, options=opcoes_traduzidas, key=f'pred_{item}')
            valor_original_en = next(key for key, value in mapa_traducoes[item].items() if value == selecao_pt)
            dicionario_features[f'{item}_{valor_original_en}'] = 1

    # ----------------------------------------------------------

    if st.button('Prever Valor do Imóvel'):
        dicionario_features.update(x_numericos)
        dicionario_features.update(x_tf)
        dicionario_features['ano'] = ano_selecionado
        valores_x = pd.DataFrame(dicionario_features, index=[0])
        try:
            modelo = joblib.load('modelo.joblib')
            dados = pd.read_csv('dados_amostra.csv')
            colunas = list(dados.columns)[1:-1]
            valores_x = valores_x[colunas]
            preco = modelo.predict(valores_x)
            st.success(f"Valor Previsto do Imóvel: R$ {preco[0]:.2f}")
        except FileNotFoundError:
            st.error("Arquivo 'modelo.joblib' ou 'colunas.joblib' não encontrado.")
        except Exception as e:
            st.error(f"Ocorreu um erro na previsão: {e}")

# streamlit run deploy_projeto_airbnb_v4.py

