import chess.pgn
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Tahtadaki taşların gelişim durumuna göre özellikleri çıkarma
def get_board_features(board):
    white_pieces = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.WHITE)) + \
                    len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.WHITE)) + \
                    len(board.pieces(chess.QUEEN, chess.WHITE))
    black_pieces = len(board.pieces(chess.PAWN, chess.BLACK)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) + \
                    len(board.pieces(chess.BISHOP, chess.BLACK)) + len(board.pieces(chess.ROOK, chess.BLACK)) + \
                    len(board.pieces(chess.QUEEN, chess.BLACK))

    # Merkez kontrolü (özellikle ortadaki 4 kareye bakılır: d4, d5, e4, e5)
    white_center_control = len([sq for sq in [chess.D4, chess.D5, chess.E4, chess.E5] if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE])
    black_center_control = len([sq for sq in [chess.D4, chess.D5, chess.E4, chess.E5] if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK])

    # Şahın güvenliği: şahın etrafındaki tehditler
    white_king_threats = len([sq for sq in chess.SQUARES if board.is_attacked_by(chess.WHITE, sq) and board.piece_at(sq) and board.piece_at(sq).piece_type == chess.KING])
    black_king_threats = len([sq for sq in chess.SQUARES if board.is_attacked_by(chess.BLACK, sq) and board.piece_at(sq) and board.piece_at(sq).piece_type == chess.KING])

    # Piyon yapısı: piyonlar arasında bağlı olanlar (bir zincir oluşturuyor mu)
    white_pawn_structure = len(board.pieces(chess.PAWN, chess.WHITE))  # Sadece piyon sayısı örnek
    black_pawn_structure = len(board.pieces(chess.PAWN, chess.BLACK))

    return [white_pieces, black_pieces, white_center_control, black_center_control, white_king_threats, black_king_threats, white_pawn_structure, black_pawn_structure]

# PGN dosyasını okuma fonksiyonu
def parse_pgn(pgn_file_path):
    games = []
    with open(pgn_file_path, "r") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            board = chess.Board()
            # Tahtadaki taşların gelişim durumunu almak
            features = get_board_features(board)

            # Oyun sonucunu al (1: Beyaz kazandı, -1: Siyah kazandı, 0: Beraberlik)
            result = game.headers['Result']
            if result == '1-0':
                label = 2  # Beyaz kazandı
                winner = "Beyaz"
            elif result == '0-1':
                label = 0  # Siyah kazandı
                winner = "Siyah"
            else:
                label = 1  # Beraberlik
                winner = "Beraberlik"

            # Özellikleri ve etiketi listeye ekle
            games.append(features + [label, winner])

    return games

# Modeli eğitme fonksiyonu
def train_model(games_df):
    # Özellikler ve etiketler
    X = games_df.drop(['Result', 'Winner'], axis=1)  # Winner sütununu da kaldırıyoruz
    y = games_df['Result']

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost modelini oluşturma
    model = xgb.XGBClassifier()

    # Modeli eğitme
    model.fit(X_train, y_train)

    # Modelin test setindeki performansını değerlendirme
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test seti doğruluk oranı: {accuracy}")

    return model

# Yeni bir PGN dosyasındaki tek bir oyunu tahmin etmek için fonksiyon
def predict_game_result(model, pgn_file_path):
    # PGN dosyasını açma ve ilk oyunu okuma
    with open(pgn_file_path, "r") as pgn_file:
        game = chess.pgn.read_game(pgn_file)

    board = chess.Board()
    # Tahtadaki taşların gelişim durumunu almak
    features = get_board_features(board)

    # Model ile tahmin yap
    prediction = model.predict([features])

    # Tahmin sonucunu yazdır
    result = "Beyaz kazandı" if prediction == 2 else "Siyah kazandı" if prediction == 0 else "Beraberlik"
    print(f"Bu oyunda: {result}")

# Ana fonksiyon
def main():
    # PGN dosyasının yolu
    pgn_file_path = "C:\\Users\\vural\\PycharmProjects\\pythonProject2\\.venv\\lichess_db_standard_rated_2015-08.pgn"

    # PGN dosyasını işleyerek oyun verilerini al
    games = parse_pgn(pgn_file_path)

    # Veri setini oluşturma
    games_df = pd.DataFrame(games, columns=['White_Pieces', 'Black_Pieces', 'White_Center_Control', 'Black_Center_Control',
                                             'White_King_Threats', 'Black_King_Threats', 'White_Pawn_Structure',
                                             'Black_Pawn_Structure', 'Result', 'Winner'])

    # Etiketlerin dağılımını kontrol etme
    print("Etiketlerin Dağılımı:")
    print(games_df['Result'].value_counts())

    # Oyun sonuçlarını yazdırma
    print("\nOyun Sonuçları:")
    for index, row in games_df.iterrows():
        print(f"Oyun {index + 1}: {row['Winner']} kazandı")

    # Modeli eğitme
    model = train_model(games_df)

    # Test için yeni bir oyun dosyasında tahmin yap
    test_pgn_file_path = "C:\\Users\\vural\\PycharmProjects\\pythonProject2\\.venv\\chess.pgn"  # Test dosyasının yolu
    predict_game_result(model, test_pgn_file_path)

# Programı çalıştır
if __name__ == "__main__":
    main()
