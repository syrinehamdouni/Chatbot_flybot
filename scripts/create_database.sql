-- Script pour créer la base de données des réservations
CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT NOT NULL,
    email TEXT NOT NULL,
    details_vol TEXT NOT NULL,
    date_reservation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    statut TEXT DEFAULT 'En attente',
    prix_estime REAL,
    telephone TEXT,
    commentaires TEXT
);

-- Index pour améliorer les performances
CREATE INDEX IF NOT EXISTS idx_email ON reservations(email);
CREATE INDEX IF NOT EXISTS idx_date ON reservations(date_reservation);
CREATE INDEX IF NOT EXISTS idx_statut ON reservations(statut);

-- Insérer quelques données de test
INSERT OR IGNORE INTO reservations (nom, email, details_vol, prix_estime) VALUES 
('Test User', 'test@example.com', 'Tunis vers Paris, 20 juin 2024', 450),
('Demo Client', 'demo@example.com', 'Sfax vers Rome, 15 juillet 2024', 380);
