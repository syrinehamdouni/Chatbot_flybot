from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
import requests
import json
import sqlite3
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

app = Flask(__name__)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Base de données pour les réservations ---
def init_database():
    """Initialiser la base de données SQLite pour les réservations"""
    conn = sqlite3.connect('reservations.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL,
            email TEXT NOT NULL,
            details_vol TEXT NOT NULL,
            date_reservation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            statut TEXT DEFAULT 'En attente'
        )
    ''')
    
    conn.commit()
    conn.close()

def save_reservation(nom, email, details_vol):
    """Sauvegarder une réservation dans la base de données"""
    try:
        conn = sqlite3.connect('reservations.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reservations (nom, email, details_vol)
            VALUES (?, ?, ?)
        ''', (nom, email, details_vol))
        
        reservation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return reservation_id
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")
        return None

# --- Système RAG amélioré ---
class FlightRAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_and_clean_data()
        self.create_embeddings()
    
    def load_and_clean_data(self):
        """Charger et nettoyer les données pour qu'elles soient réalistes"""
        try:
            # Charger les données
            self.flights = pd.read_csv('flights.csv')
            self.locations = pd.read_csv('locations.csv')
            self.airlines = pd.read_csv('airlines.csv')
            
            # Merger les données
            self.flights = self.flights.merge(
                self.locations, 
                left_on='localisation_fk', 
                right_on='id_localisation', 
                how='left'
            )
            
            # Nettoyer et normaliser les données
            self.flights['departure_city'] = self.flights['departure_city'].str.strip().str.lower()
            self.flights['arrival_city'] = self.flights['arrival_city'].str.strip().str.lower()
            
            # **CORRECTION MAJEURE : Nettoyer les prix irréalistes**
            # Supprimer les vols avec prix = 0 ou vols annulés
            self.flights = self.flights[
                (self.flights['Prix_Final'] > 0) & 
                (self.flights['CANCELLED'] == 0)
            ].copy()
            
            # Normaliser les prix aberrants (> 15000 TND)
            self.flights.loc[self.flights['Prix_Final'] > 15000, 'Prix_Final'] = \
                self.flights.loc[self.flights['Prix_Final'] > 15000, 'Prix_Final'] / 10
            
            # Ajouter les informations des compagnies
            self.flights = self.flights.merge(
                self.airlines,
                left_on='fk_flights',
                right_on='id_airline',
                how='left'
            )
            
            # Remplir les valeurs manquantes
            self.flights['AirlineName'] = self.flights['AirlineName'].fillna('Tunisair')
            
            # **Créer des données réalistes si nécessaire**
            if len(self.flights) < 10:
                self.create_realistic_data()
            
            logger.info(f"Données nettoyées: {len(self.flights)} vols valides")
            logger.info(f"Prix min: {self.flights['Prix_Final'].min():.0f} TND")
            logger.info(f"Prix max: {self.flights['Prix_Final'].max():.0f} TND")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            self.create_realistic_data()
    
    def create_realistic_data(self):
        """Créer des données de vol réalistes"""
        realistic_flights = [
            # Vols depuis Tunis
            {'departure_city': 'tunis', 'arrival_city': 'paris', 'Prix_Final': 450, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1500, 'Nombre_Escales': 0},
            {'departure_city': 'tunis', 'arrival_city': 'rome', 'Prix_Final': 380, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1200, 'Nombre_Escales': 0},
            {'departure_city': 'tunis', 'arrival_city': 'london', 'Prix_Final': 520, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1800, 'Nombre_Escales': 1},
            {'departure_city': 'tunis', 'arrival_city': 'madrid', 'Prix_Final': 410, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1300, 'Nombre_Escales': 0},
            {'departure_city': 'tunis', 'arrival_city': 'istanbul', 'Prix_Final': 350, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1600, 'Nombre_Escales': 0},
            {'departure_city': 'tunis', 'arrival_city': 'frankfurt', 'Prix_Final': 480, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1700, 'Nombre_Escales': 1},
            {'departure_city': 'tunis', 'arrival_city': 'barcelona', 'Prix_Final': 390, 'AirlineName': 'Nouvelair', 'Distance_Vol_KM': 1250, 'Nombre_Escales': 0},
            {'departure_city': 'tunis', 'arrival_city': 'cairo', 'Prix_Final': 320, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1900, 'Nombre_Escales': 0},
            {'departure_city': 'tunis', 'arrival_city': 'dubai', 'Prix_Final': 850, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 4500, 'Nombre_Escales': 1},
        
            # Vols depuis Sfax
            {'departure_city': 'sfax', 'arrival_city': 'paris', 'Prix_Final': 470, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1550, 'Nombre_Escales': 0},
            {'departure_city': 'sfax', 'arrival_city': 'rome', 'Prix_Final': 400, 'AirlineName': 'Nouvelair', 'Distance_Vol_KM': 1250, 'Nombre_Escales': 0},
            {'departure_city': 'sfax', 'arrival_city': 'frankfurt', 'Prix_Final': 480, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1700, 'Nombre_Escales': 1},
        
            # Vols depuis Monastir
            {'departure_city': 'monastir', 'arrival_city': 'paris', 'Prix_Final': 460, 'AirlineName': 'Nouvelair', 'Distance_Vol_KM': 1520, 'Nombre_Escales': 0},
            {'departure_city': 'monastir', 'arrival_city': 'london', 'Prix_Final': 540, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1850, 'Nombre_Escales': 1},
            {'departure_city': 'monastir', 'arrival_city': 'berlin', 'Prix_Final': 490, 'AirlineName': 'Nouvelair', 'Distance_Vol_KM': 1750, 'Nombre_Escales': 0},
        
            # Vols depuis Djerba
            {'departure_city': 'djerba', 'arrival_city': 'paris', 'Prix_Final': 480, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1580, 'Nombre_Escales': 0},
            {'departure_city': 'djerba', 'arrival_city': 'rome', 'Prix_Final': 420, 'AirlineName': 'Nouvelair', 'Distance_Vol_KM': 1280, 'Nombre_Escales': 0},
            {'departure_city': 'djerba', 'arrival_city': 'frankfurt', 'Prix_Final': 500, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1720, 'Nombre_Escales': 1},
        
            # Vols retour
            {'departure_city': 'paris', 'arrival_city': 'tunis', 'Prix_Final': 460, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1500, 'Nombre_Escales': 0},
            {'departure_city': 'rome', 'arrival_city': 'tunis', 'Prix_Final': 390, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1200, 'Nombre_Escales': 0},
            {'departure_city': 'london', 'arrival_city': 'tunis', 'Prix_Final': 530, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1800, 'Nombre_Escales': 1},
            {'departure_city': 'istanbul', 'arrival_city': 'tunis', 'Prix_Final': 360, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1600, 'Nombre_Escales': 0},
            {'departure_city': 'frankfurt', 'arrival_city': 'tunis', 'Prix_Final': 490, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1700, 'Nombre_Escales': 0},
            {'departure_city': 'cairo', 'arrival_city': 'tunis', 'Prix_Final': 330, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1900, 'Nombre_Escales': 0},
        ]
    
        self.flights = pd.DataFrame(realistic_flights)
        logger.info(f"Données réalistes créées: {len(self.flights)} vols")
    
    def create_embeddings(self):
        """Créer les embeddings pour la recherche sémantique"""
        self.flight_descriptions = []
        for _, row in self.flights.iterrows():
            description = f"Vol de {row['departure_city']} à {row['arrival_city']} avec {row['AirlineName']} pour {row['Prix_Final']} TND"
            self.flight_descriptions.append(description)
        
        if self.flight_descriptions:
            self.embeddings = self.model.encode(self.flight_descriptions)
            logger.info(f"Embeddings créés pour {len(self.flight_descriptions)} vols")
    
    def semantic_search(self, query, top_k=3):
        """Recherche sémantique dans les vols"""
        if not hasattr(self, 'embeddings') or len(self.embeddings) == 0:
            return []
            
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.2:  # Seuil de similarité réduit
                results.append({
                    'flight': self.flights.iloc[idx],
                    'description': self.flight_descriptions[idx],
                    'similarity': similarities[idx]
                })
        
        return results

# Initialiser le système
init_database()
flight_rag = FlightRAG()

# --- Configuration OLLAMA ---
OLLAMA_API_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'deepseek-r1:latest'

def query_ollama(prompt):
    """Interroger OLLAMA avec gestion d'erreur"""
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get('response', '')
    except Exception as e:
        logger.warning(f"OLLAMA non disponible: {e}")
    
    return None

def normalize_city_name(city):
    """Normaliser les noms de villes avec plus de variantes"""
    city_mapping = {
        # Tunisie
        'tunisia': 'tunis', 'tunisie': 'tunis', 'tunis': 'tunis', 'tunes': 'tunis',
        'sfax': 'sfax', 'safax': 'sfax', 'safaqis': 'sfax',
        'monastir': 'monastir', 'monastire': 'monastir', 'monaster': 'monastir',
        'djerba': 'djerba', 'jerba': 'djerba', 'gerba': 'djerba', 'djarba': 'djerba',
        'tozeur': 'tozeur', 'touzeur': 'tozeur', 'tozer': 'tozeur',
        
        # Europe
        'rome': 'rome', 'roma': 'rome', 'italie': 'rome', 'italy': 'rome',
        'paris': 'paris', 'france': 'paris', 'pari': 'paris',
        'london': 'london', 'londres': 'london', 'angleterre': 'london', 'uk': 'london',
        'madrid': 'madrid', 'espagne': 'madrid', 'spain': 'madrid',
        'berlin': 'berlin', 'allemagne': 'berlin', 'germany': 'berlin',
        'frankfurt': 'frankfurt', 'francfort': 'frankfurt', 'frankfort': 'frankfurt',
        'istanbul': 'istanbul', 'turquie': 'istanbul', 'turkey': 'istanbul',
        'nice': 'nice', 'nizza': 'nice', 'nis': 'nice',
        'barcelona': 'barcelona', 'barcelone': 'barcelona', 'barça': 'barcelona',
        
        # Autres
        'cairo': 'cairo', 'le caire': 'cairo', 'egypte': 'cairo', 'egypt': 'cairo',
        'dubai': 'dubai', 'emirats': 'dubai', 'uae': 'dubai', 'dubaï': 'dubai'
    }
    
    if not city:
        return ""
        
    normalized = city.lower().strip()
    return city_mapping.get(normalized, normalized)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data.get('question', '').lower()
    
    try:
        # **NOUVELLES QUESTIONS BASÉES SUR LES DONNÉES CSV**
        
        # Questions sur les escales
        if re.search(r'(escale|stopover|connection|correspondance)', question):
            if re.search(r'(sans escale|direct|non.stop)', question):
                direct_flights = flight_rag.flights[flight_rag.flights['Nombre_Escales'] == 0]
                if not direct_flights.empty:
                    count = len(direct_flights)
                    min_price = direct_flights['Prix_Final'].min()
                    max_price = direct_flights['Prix_Final'].max()
                    return jsonify({'answer': f"✈️ {count} vols directs disponibles (sans escale)\n💰 Prix: {min_price:.0f} - {max_price:.0f} TND"})
            else:
                escale_stats = flight_rag.flights['Nombre_Escales'].value_counts().sort_index()
                result = "🔄 Statistiques des escales:\n"
                for escales, count in escale_stats.items():
                    if escales == 0:
                        result += f"• Vols directs: {count} vols\n"
                    else:
                        result += f"• {escales} escale(s): {count} vols\n"
                return jsonify({'answer': result})
        
        # Questions sur les distances
        elif re.search(r'(distance|km|kilomètre|far|loin)', question):
            if re.search(r'(plus.*long|longest|maximum)', question):
                longest = flight_rag.flights.loc[flight_rag.flights['Distance_Vol_KM'].idxmax()]
                return jsonify({'answer': f"🌍 Vol le plus long: {longest['departure_city'].capitalize()} → {longest['arrival_city'].capitalize()}\n📏 Distance: {longest['Distance_Vol_KM']} km\n💰 Prix: {longest['Prix_Final']:.0f} TND"})
            elif re.search(r'(plus.*court|shortest|minimum)', question):
                shortest = flight_rag.flights.loc[flight_rag.flights['Distance_Vol_KM'].idxmin()]
                return jsonify({'answer': f"🌍 Vol le plus court: {shortest['departure_city'].capitalize()} → {shortest['arrival_city'].capitalize()}\n📏 Distance: {shortest['Distance_Vol_KM']} km\n💰 Prix: {shortest['Prix_Final']:.0f} TND"})
            else:
                avg_distance = flight_rag.flights['Distance_Vol_KM'].mean()
                min_distance = flight_rag.flights['Distance_Vol_KM'].min()
                max_distance = flight_rag.flights['Distance_Vol_KM'].max()
                return jsonify({'answer': f"📏 Distances des vols:\n• Minimum: {min_distance} km\n• Maximum: {max_distance} km\n• Moyenne: {avg_distance:.0f} km"})
        
        # Questions sur les compagnies aériennes spécifiques
        elif re.search(r'(compagnie|airline|tunisair|nouvelair)', question):
            if 'tunisair' in question:
                tunisair_flights = flight_rag.flights[flight_rag.flights['AirlineName'].str.lower() == 'tunisair']
                if not tunisair_flights.empty:
                    count = len(tunisair_flights)
                    min_price = tunisair_flights['Prix_Final'].min()
                    max_price = tunisair_flights['Prix_Final'].max()
                    avg_price = tunisair_flights['Prix_Final'].mean()
                    destinations = tunisair_flights['arrival_city'].unique()
                    return jsonify({'answer': f"✈️ Tunisair:\n• {count} vols disponibles\n• Prix: {min_price:.0f} - {max_price:.0f} TND (moy: {avg_price:.0f})\n• Destinations: {', '.join([d.capitalize() for d in destinations[:5]])}"})
            elif 'nouvelair' in question:
                nouvelair_flights = flight_rag.flights[flight_rag.flights['AirlineName'].str.lower() == 'nouvelair']
                if not nouvelair_flights.empty:
                    count = len(nouvelair_flights)
                    min_price = nouvelair_flights['Prix_Final'].min()
                    max_price = nouvelair_flights['Prix_Final'].max()
                    avg_price = nouvelair_flights['Prix_Final'].mean()
                    destinations = nouvelair_flights['arrival_city'].unique()
                    return jsonify({'answer': f"✈️ Nouvelair:\n• {count} vols disponibles\n• Prix: {min_price:.0f} - {max_price:.0f} TND (moy: {avg_price:.0f})\n• Destinations: {', '.join([d.capitalize() for d in destinations[:5]])}"})
            else:
                airlines_stats = flight_rag.flights['AirlineName'].value_counts()
                result = "🏢 Compagnies aériennes:\n"
                for airline, count in airlines_stats.items():
                    avg_price = flight_rag.flights[flight_rag.flights['AirlineName'] == airline]['Prix_Final'].mean()
                    result += f"• {airline}: {count} vols (moy: {avg_price:.0f} TND)\n"
                return jsonify({'answer': result})
        
        # Questions sur les vols annulés
        elif re.search(r'(annulé|cancelled|cancel)', question):
            cancelled_flights = flight_rag.flights[flight_rag.flights['CANCELLED'] == 1] if 'CANCELLED' in flight_rag.flights.columns else pd.DataFrame()
            total_flights = len(flight_rag.flights)
            if not cancelled_flights.empty:
                cancelled_count = len(cancelled_flights)
                cancellation_rate = (cancelled_count / total_flights) * 100
                return jsonify({'answer': f"❌ Vols annulés: {cancelled_count} sur {total_flights} ({cancellation_rate:.1f}%)"})
            else:
                return jsonify({'answer': f"✅ Aucun vol annulé dans notre base de données ({total_flights} vols actifs)"})
        
        # Questions sur les taxes
        elif re.search(r'(taxe|tax|frais)', question):
            if 'Taxe_Price' in flight_rag.flights.columns:
                avg_tax = flight_rag.flights['Taxe_Price'].mean()
                min_tax = flight_rag.flights['Taxe_Price'].min()
                max_tax = flight_rag.flights['Taxe_Price'].max()
                return jsonify({'answer': f"💳 Taxes et frais:\n• Minimum: {min_tax:.0f} TND\n• Maximum: {max_tax:.0f} TND\n• Moyenne: {avg_tax:.0f} TND"})
        
        # Questions sur les routes spécifiques avec plus de détails
        elif re.search(r'(route|trajet|itinéraire)', question):
            routes = flight_rag.flights.groupby(['departure_city', 'arrival_city']).agg({
                'Prix_Final': ['count', 'mean', 'min'],
                'Distance_Vol_KM': 'first'
            }).round(0)
            
            result = "🗺️ Routes populaires:\n"
            for i, ((dep, arr), data) in enumerate(routes.head(5).iterrows()):
                count = int(data[('Prix_Final', 'count')])
                avg_price = int(data[('Prix_Final', 'mean')])
                min_price = int(data[('Prix_Final', 'min')])
                distance = int(data[('Distance_Vol_KM', 'first')])
                result += f"• {dep.capitalize()} → {arr.capitalize()}: {count} vols, {distance}km, à partir de {min_price} TND\n"
            
            return jsonify({'answer': result})
        
        # Questions sur les statistiques générales
        elif re.search(r'(statistique|stats|combien.*total|nombre.*vol)', question):
            total_flights = len(flight_rag.flights)
            total_routes = len(flight_rag.flights.groupby(['departure_city', 'arrival_city']))
            total_cities = len(set(flight_rag.flights['departure_city'].unique()) | set(flight_rag.flights['arrival_city'].unique()))
            avg_price = flight_rag.flights['Prix_Final'].mean()
            avg_distance = flight_rag.flights['Distance_Vol_KM'].mean()
            
            return jsonify({'answer': f"📊 Statistiques générales:\n• {total_flights} vols disponibles\n• {total_routes} routes différentes\n• {total_cities} villes desservies\n• Prix moyen: {avg_price:.0f} TND\n• Distance moyenne: {avg_distance:.0f} km"})
        
        # Questions sur les prix par tranche
        elif re.search(r'(budget|économique|cher|prix.*entre)', question):
            if re.search(r'(moins.*500|budget|économique)', question):
                cheap_flights = flight_rag.flights[flight_rag.flights['Prix_Final'] < 500]
                if not cheap_flights.empty:
                    count = len(cheap_flights)
                    routes = cheap_flights.groupby(['departure_city', 'arrival_city'])['Prix_Final'].min().head(5)
                    result = f"💰 {count} vols économiques (< 500 TND):\n"
                    for (dep, arr), price in routes.items():
                        result += f"• {dep.capitalize()} → {arr.capitalize()}: {price:.0f} TND\n"
                    return jsonify({'answer': result})
            elif re.search(r'(plus.*1000|luxe|premium)', question):
                expensive_flights = flight_rag.flights[flight_rag.flights['Prix_Final'] > 1000]
                if not expensive_flights.empty:
                    count = len(expensive_flights)
                    routes = expensive_flights.groupby(['departure_city', 'arrival_city'])['Prix_Final'].min().head(5)
                    result = f"💎 {count} vols premium (> 1000 TND):\n"
                    for (dep, arr), price in routes.items():
                        result += f"• {dep.capitalize()} → {arr.capitalize()}: {price:.0f} TND\n"
                    return jsonify({'answer': result})
        
        # Villes disponibles
        if re.search(r'(cities.*fly\s+from|departure\s+cities|from\s+cities|villes.*départ|quelles villes.*partir)', question):
            cities = flight_rag.flights['departure_city'].unique()
            cities_str = ', '.join([city.capitalize() for city in cities])
            return jsonify({'answer': f"🛫 Villes de départ disponibles: {cities_str}"})
        
        elif re.search(r'(cities.*fly\s+to|arrival\s+cities|to\s+cities|villes.*arrivée|quelles villes.*visiter)', question):
            cities = flight_rag.flights['arrival_city'].unique()
            cities_str = ', '.join([city.capitalize() for city in cities])
            return jsonify({'answer': f"🛬 Destinations disponibles: {cities_str}"})
        
        # Vol le moins cher
        elif re.search(r'(cheapest|moins cher|prix.*bas|économique)', question):
            if not flight_rag.flights.empty:
                cheapest = flight_rag.flights.loc[flight_rag.flights['Prix_Final'].idxmin()]
                answer = f"💰 Le vol le moins cher: {cheapest['departure_city'].capitalize()} → {cheapest['arrival_city'].capitalize()} pour {cheapest['Prix_Final']:.0f} TND avec {cheapest['AirlineName']}"
                return jsonify({'answer': answer})
        
        # Vol le plus cher
        elif re.search(r'(most expensive|plus cher|prix.*élevé)', question):
            if not flight_rag.flights.empty:
                expensive = flight_rag.flights.loc[flight_rag.flights['Prix_Final'].idxmax()]
                answer = f"💎 Le vol le plus cher: {expensive['departure_city'].capitalize()} → {expensive['arrival_city'].capitalize()} pour {expensive['Prix_Final']:.0f} TND avec {expensive['AirlineName']}"
                return jsonify({'answer': answer})
        
        # **CORRECTION MAJEURE : Recherche de vol spécifique**
        elif re.search(r'(vol.*de|from|flight.*from)\s+([a-zA-Zàâäéèêëïîôöùûüÿç\s]+)\s+(à|vers|to)\s+([a-zA-Zàâäéèêëïîôöùûüÿç\s]+)', question):
            match = re.search(r'(vol.*de|from|flight.*from)\s+([a-zA-Zàâäéèêëïîôöùûüÿç\s]+)\s+(à|vers|to)\s+([a-zA-Zàâäéèêëïîôöùûüÿç\s]+)', question)
            if match:
                from_city = normalize_city_name(match.group(2).strip())
                to_city = normalize_city_name(match.group(4).strip())
                
                # Ajouter des logs pour le débogage
                logger.info(f"Recherche de vol: de '{from_city}' à '{to_city}'")
                
                # Rechercher les vols correspondants avec une approche plus flexible
                available = flight_rag.flights[
                    (flight_rag.flights['departure_city'].str.lower() == from_city.lower()) &
                    (flight_rag.flights['arrival_city'].str.lower() == to_city.lower())
                ]
                
                # Si aucun vol n'est trouvé, essayer avec les données réalistes
                if available.empty:
                    logger.info(f"Aucun vol trouvé dans les données CSV, utilisation des données réalistes")
                    # Créer des données réalistes pour les routes populaires
                    realistic_flights = [
                        {'departure_city': 'tunis', 'arrival_city': 'rome', 'Prix_Final': 380, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1200, 'Nombre_Escales': 0},
                        {'departure_city': 'tunis', 'arrival_city': 'paris', 'Prix_Final': 450, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1500, 'Nombre_Escales': 0},
                        {'departure_city': 'tunis', 'arrival_city': 'london', 'Prix_Final': 520, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1800, 'Nombre_Escales': 1},
                        {'departure_city': 'tunis', 'arrival_city': 'madrid', 'Prix_Final': 410, 'AirlineName': 'Nouvelair', 'Distance_Vol_KM': 1300, 'Nombre_Escales': 0},
                        {'departure_city': 'tunis', 'arrival_city': 'istanbul', 'Prix_Final': 350, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1600, 'Nombre_Escales': 0},
                        {'departure_city': 'tunis', 'arrival_city': 'frankfurt', 'Prix_Final': 480, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1700, 'Nombre_Escales': 1},
                        {'departure_city': 'tunis', 'arrival_city': 'barcelona', 'Prix_Final': 390, 'AirlineName': 'Nouvelair', 'Distance_Vol_KM': 1250, 'Nombre_Escales': 0},
                        {'departure_city': 'tunis', 'arrival_city': 'cairo', 'Prix_Final': 320, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 1900, 'Nombre_Escales': 0},
                        {'departure_city': 'tunis', 'arrival_city': 'dubai', 'Prix_Final': 850, 'AirlineName': 'Tunisair', 'Distance_Vol_KM': 4500, 'Nombre_Escales': 1},
                    ]
                    
                    # Filtrer les vols réalistes pour la recherche actuelle
                    realistic_df = pd.DataFrame(realistic_flights)
                    available = realistic_df[
                        (realistic_df['departure_city'].str.lower() == from_city.lower()) &
                        (realistic_df['arrival_city'].str.lower() == to_city.lower())
                    ]
                
                if not available.empty:
                    results = []
                    for _, row in available.head(3).iterrows():
                        escales_text = "direct" if row.get('Nombre_Escales', 0) == 0 else f"{row.get('Nombre_Escales', 0)} escale(s)"
                        results.append(f"✈️ {row['departure_city'].capitalize()} → {row['arrival_city'].capitalize()}: {row['Prix_Final']:.0f} TND ({row['AirlineName']}, {escales_text}, {row.get('Distance_Vol_KM', 0)}km)")
                    
                    answer = f"🔍 Vols trouvés:\n" + "\n".join(results)
                    return jsonify({'answer': answer})
                else:
                    # Suggérer des destinations alternatives
                    all_destinations = flight_rag.flights['arrival_city'].unique()
                    suggestions = ", ".join([d.capitalize() for d in all_destinations[:5]])
                    return jsonify({'answer': f"❌ Aucun vol direct trouvé de {from_city.capitalize()} à {to_city.capitalize()}.\nDestinations disponibles: {suggestions}"})
        
        # Prix d'un vol spécifique (version alternative)
        elif re.search(r'prix.*vol|cost.*flight|combien.*coûte', question):
            semantic_results = flight_rag.semantic_search(question)
            if semantic_results:
                flight = semantic_results[0]['flight']
                answer = f"💰 Prix du vol {flight['departure_city'].capitalize()} → {flight['arrival_city'].capitalize()}: {flight['Prix_Final']:.0f} TND avec {flight['AirlineName']}"
                return jsonify({'answer': answer})
        
        # Offres spéciales (prix < 450 TND)
        elif re.search(r'(offer|offre|promo|reduction|pas cher)', question):
            cheap_flights = flight_rag.flights[flight_rag.flights['Prix_Final'] < 450]
            if not cheap_flights.empty:
                results = []
                for _, row in cheap_flights.head(3).iterrows():
                    results.append(f"🎯 {row['departure_city'].capitalize()} → {row['arrival_city'].capitalize()}: {row['Prix_Final']:.0f} TND")
                return jsonify({'answer': "🔥 Offres spéciales (< 450 TND):\n" + "\n".join(results)})
            return jsonify({'answer': "😔 Aucune offre spéciale disponible actuellement."})
        
        # Réservation
        elif re.search(r'(reservation|book|réserver|réservation)', question):
            return jsonify({'answer': "📝 Pour effectuer une réservation, veuillez remplir le formulaire ci-dessous avec vos informations."})
        
        # Recherche sémantique par défaut
        semantic_results = flight_rag.semantic_search(question)
        if semantic_results:
            flight = semantic_results[0]['flight']
            answer = f"✈️ Vol suggéré: {flight['departure_city'].capitalize()} → {flight['arrival_city'].capitalize()} pour {flight['Prix_Final']:.0f} TND avec {flight['AirlineName']}"
            return jsonify({'answer': answer})
        
        # Réponse par défaut améliorée
        return jsonify({'answer': "🤔 Je ne comprends pas votre question. Essayez:\n• 'Vol de Tunis à Rome'\n• 'Vols sans escale'\n• 'Statistiques des vols'\n• 'Compagnies aériennes'\n• 'Routes disponibles'\n• 'Vols annulés'"})
        
    except Exception as e:
        logger.error(f"Erreur dans le chatbot: {e}")
        return jsonify({'answer': "😔 Désolé, une erreur s'est produite. Veuillez réessayer."})

@app.route('/save_reservation', methods=['POST'])
def save_reservation_route():
    """Route pour sauvegarder les réservations"""
    data = request.get_json()
    nom = data.get('nom', '')
    email = data.get('email', '')
    details_vol = data.get('details_vol', '')
    
    if nom and email and details_vol:
        reservation_id = save_reservation(nom, email, details_vol)
        if reservation_id:
            return jsonify({
                'success': True, 
                'message': f"✅ Réservation #{reservation_id} enregistrée avec succès!",
                'reservation_id': reservation_id
            })
    
    return jsonify({'success': False, 'message': "❌ Erreur lors de l'enregistrement"})

@app.route('/admin/reservations')
def view_reservations():
    """Route admin pour voir les réservations"""
    try:
        conn = sqlite3.connect('reservations.db')
        df = pd.read_sql_query("SELECT * FROM reservations ORDER BY date_reservation DESC", conn)
        conn.close()
        
        return f"<h1>Réservations ({len(df)})</h1>" + df.to_html()
    except Exception as e:
        return f"Erreur: {e}"

if __name__ == '__main__':
    app.run(debug=True)
