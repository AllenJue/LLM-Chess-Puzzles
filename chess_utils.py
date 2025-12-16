"""
Chess Utilities Module

This module contains utility functions for chess-related operations including
PGN processing, move extraction, and board state management.
"""

import re
import requests
import chess
import chess.pgn
from io import StringIO
from typing import Optional, Tuple, List

SAN_SUB_PATTERN = r'(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[a-h]x?[a-h][1-8][+#]?|[a-h][1-8][+#]?)'


def extract_game_id_and_move(url: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract the game ID (8-chars) and move number (#ply) from any Lichess URL variant:
      - https://lichess.org/4MWQCxQ6#32
      - https://lichess.org/4MWQCxQ6/black#32
      - https://lichess.org/4MWQCxQ6/white#10
    
    Args:
        url (str): Lichess game URL
        
    Returns:
        Tuple[Optional[str], Optional[int]]: (game_id, move_number)
    """
    # Match 8-character game ID
    match = re.search(r"lichess\.org/([A-Za-z0-9]{8})", url)
    game_id = match.group(1) if match else None

    # Extract move number (if present after "#")
    move_match = re.search(r"#([0-9]+)", url)
    move_number = int(move_match.group(1)) if move_match else None

    return game_id, move_number


def get_clean_pgn(game_id: str) -> str:
    """
    Fetch PGN from the Lichess API without evals or clocks.
    
    Args:
        game_id (str): Lichess game ID
        
    Returns:
        str: Clean PGN string
        
    Raises:
        ValueError: If API request fails
    """
    url = f"https://lichess.org/game/export/{game_id}"
    params = {"evals": "false", "clocks": "false"}
    headers = {"Accept": "application/x-chess-pgn"}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        return res.text
    raise ValueError(f"Failed to fetch PGN for {game_id}: {res.status_code}")


def truncate_pgn_at_move(pgn_text: str, move_number: int) -> str:
    """
    Truncate a PGN string at a given ply (half-move) count.
    Example: move_number=32 means 16 moves by each side.
    
    Args:
        pgn_text (str): Full PGN string
        move_number (int): Number of plies to keep
        
    Returns:
        str: Truncated PGN string
    """
    game = chess.pgn.read_game(StringIO(pgn_text))
    board = game.board()

    truncated_game = chess.pgn.Game()
    truncated_game.headers = game.headers.copy()

    node = truncated_game
    move_count = 0
    for move in game.mainline_moves():
        move_count += 1
        if move_count > move_number:
            break
        node = node.add_main_variation(move)
        board.push(move)

    # Return truncated PGN
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return truncated_game.accept(exporter)


def get_partial_pgn_from_url(url: str) -> Tuple[str, Optional[int]]:
    """
    Combine everything: extract ID, download, and truncate PGN.
    
    Args:
        url (str): Lichess game URL
        
    Returns:
        Tuple[str, Optional[int]]: (pgn, move_number)
        
    Raises:
        ValueError: If URL is invalid
    """
    game_id, move_num = extract_game_id_and_move(url)
    if not game_id:
        raise ValueError(f"Invalid Lichess URL: {url}")

    print(f"Fetching PGN for {game_id} up to move #{move_num}")
    pgn = get_clean_pgn(game_id)

    if move_num:
        pgn = truncate_pgn_at_move(pgn, move_num)

    return pgn, move_num


def build_chess_prompts(pgn_partial: str) -> Tuple[str, str]:
    """
    Build system and user prompts for the language model from a PGN partial.

    Args:
        pgn_partial (str): A PGN string (possibly with metadata).

    Returns:
        Tuple[str, str]: (system_prompt, user_prompt)
    """
    system_prompt = """You are a chess grandmaster.
You will be given a partially completed game.
After seeing it, you should repeat the ENTIRE GAME and then give the next THREE moves.
After repeating the game, immediately continue by listing those moves in order on a single line, separated by spaces, starting with the side to move now.
Use standard algebraic notation, e.g. "e4" or "Rdf8" or "R1a3".
ALWAYS repeat the entire representation of the game so far.
NEVER explain your choice.
"""
    pgn_io = StringIO(pgn_partial)
    game = chess.pgn.read_game(pgn_io)

    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_without_metadata = game.accept(exporter)

    # Remove result suffix from PGN if present
    result_pattern = r'\s*(1-0|0-1|1/2-1/2|\*)\s*$'
    clean_pgn = re.sub(result_pattern, '', pgn_without_metadata)

    user_prompt = clean_pgn.strip()

    return system_prompt, user_prompt


def get_clean_pgn_from_pgn(pgn: str) -> str:
    """
    Clean PGN by removing metadata and result suffixes.
    
    Args:
        pgn (str): Raw PGN string
        
    Returns:
        str: Clean PGN string
    """
    pgn_io = StringIO(pgn)
    game = chess.pgn.read_game(pgn_io)

    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_without_metadata = game.accept(exporter)

    # Remove result suffix from PGN if present
    result_pattern = r'\s*(1-0|0-1|1/2-1/2|\*)\s*$'
    clean_pgn = re.sub(result_pattern, '', pgn_without_metadata)
    return clean_pgn


def san_to_uci(current_fen: str, san_move: str) -> Optional[str]:
    """
    Convert SAN move to UCI move given the current board position in FEN.

    Args:
        current_fen (str): Current board position in FEN notation.
        san_move (str): Move in SAN notation (e.g. 'Qe1#', 'Kxe1', 'e4').

    Returns:
        Optional[str]: UCI move string (e.g. 'e2e4', 'e1e2'), or None if invalid.
    """
    board = chess.Board(current_fen)
    # Remove check/mate symbols from SAN move (like # or +)
    clean_san = san_move.rstrip('#+')
    print(f"Converting SAN '{san_move}' -> '{clean_san}' on board FEN: {current_fen}")
    try:
        move = board.parse_san(clean_san)
        print(f"Parsed move: {move}")
        uci_move = move.uci()
        print(f"UCI move: {uci_move}")
        return uci_move
    except ValueError as e:
        # Invalid SAN, cannot convert
        print(f"ValueError converting SAN '{clean_san}': {e}")
        return None


def extract_plan_sans(
    response_text: str,
    primary_san: Optional[str] = None,
    max_plies: Optional[int] = None
) -> List[str]:
    """
    Extract a sequence of planned SAN moves from a model response.

    Args:
        response_text (str): Raw model response text.
        primary_san (Optional[str]): The primary SAN move that precedes the plan.
        max_plies (Optional[int]): Maximum number of plies to return.

    Returns:
        List[str]: Ordered list of SAN moves from the plan section.
    """
    if not response_text or not isinstance(response_text, str):
        return []

    text = response_text.strip()
    text = re.sub(r'\{.*?\}|\(.*?\)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)

    san_pattern = re.compile(SAN_SUB_PATTERN, flags=re.IGNORECASE)
    matches = list(san_pattern.finditer(text))
    if not matches:
        return []

    def normalize(s: str) -> str:
        return re.sub(r'[+#?!]+$', '', s.strip())

    start_index = 0
    if primary_san:
        norm_primary = normalize(primary_san)
        last_idx = None
        for idx, match in enumerate(matches):
            if normalize(match.group(0)) == norm_primary:
                last_idx = idx
        if last_idx is not None:
            start_index = last_idx + 1
        else:
            start_index = len(matches)

    plan_tokens = [matches[i].group(0).strip() for i in range(start_index, len(matches))]
    if max_plies is not None:
        plan_tokens = plan_tokens[:max_plies]
    return plan_tokens


def extract_predicted_move(response_text: str, current_turn_number: Optional[int] = None, 
                         is_white_to_move: bool = True) -> Optional[str]:
    """
    Robust SAN extractor with preference rules:
      1) explicit labelled predictions like "Predicted move san: Qe1#"
      2) SAN immediately followed by game result (e.g. "Qe1# 0-1")
      3) explicit numbered moves for the provided current_turn_number ("22..." or "22.")
      4) SANs whose nearest preceding move-number <= current_turn_number
      5) ellipsis-based heuristics and safe fallbacks
    
    Args:
        response_text (str): Model response text
        current_turn_number (Optional[int]): Current turn number
        is_white_to_move (bool): Whether it's white's turn to move
        
    Returns:
        Optional[str]: Extracted SAN move or None
    """
    if not response_text or not isinstance(response_text, str):
        return None

    # Debug output for chess game engine
    print(f"<debug> : extract_predicted_move called with:")
    print(f"<debug> :   current_turn_number: {current_turn_number}")
    print(f"<debug> :   is_white_to_move: {is_white_to_move}")
    print(f"<debug> :   response_text: {repr(response_text[:200])}...")

    text = response_text.strip()
    # Collapse multiple spaces but preserve newlines for move number patterns
    text = re.sub(r'[ \t]+', ' ', text)  # collapse spaces and tabs but not newlines
    text = re.sub(r'Model output:|Predicted Response:|Completion\(.*?\):?', '', text, flags=re.I)
    # keep result tokens for detection (we'll remove elsewhere as needed)
    text = re.sub(r'\{.*?\}|\(.*?\)', ' ', text)  # remove comments
    text = text.strip()
    
    print(f"<debug> :   processed_text: {repr(text[:200])}...")

    # --- 1) explicit labelled prediction e.g. "Predicted move san: Qe1#" ---
    labelled = re.search(r'predicted\s*move\s*(?:san)?\s*[:\-]\s*(' + SAN_SUB_PATTERN + r')', text, flags=re.I)
    if labelled:
        result = labelled.group(1).strip()
        print(f"<debug> :   found labelled prediction: {repr(result)}")
        return result

    # --- 2a) Sequential parsing with move numbers to determine exact ply ---
    # This should happen BEFORE rule #2 to avoid matching moves from hallucinated game endings
    sequential_pattern = re.compile(r'(\d+\.\.\.)|(\d+\.)|(' + SAN_SUB_PATTERN + r')', flags=re.I)
    sequential_moves: list[tuple[str, Optional[int], Optional[bool]]] = []
    current_marker: Optional[int] = None
    expect_white: Optional[bool] = None
    for token_match in sequential_pattern.finditer(text):
        token = token_match.group(0)
        if re.fullmatch(r'\d+\.\.\.', token):
            current_marker = int(token.split('.')[0])
            expect_white = False
            continue
        if re.fullmatch(r'\d+\.', token):
            current_marker = int(token[:-1])
            expect_white = True
            continue
        # SAN token
        san_token = token.strip()
        sequential_moves.append((san_token, current_marker, expect_white))
        if expect_white is None:
            continue
        if expect_white:
            expect_white = False
        else:
            expect_white = True
            if current_marker is not None:
                current_marker += 1

    if current_turn_number is not None:
        desired_turn = int(current_turn_number)
        desired_side = bool(is_white_to_move)
        for san_token, marker, side in reversed(sequential_moves):
            if marker == desired_turn and side is not None and side == desired_side:
                print(f"<debug> :   sequential match for turn {desired_turn} ({'white' if desired_side else 'black'}): {san_token}")
                return san_token

    # --- 2) SAN followed by a result token (likely a final predicted move) ---
    # Only match if sequential parsing didn't find a match for the current turn
    # This prevents matching moves from hallucinated game endings when we have proper move numbering
    # We'll check this after rule 3 (explicit numbered moves) to be safe

    # --- 2b) Handle trailing non-numbered lines (models often append future moves without move numbers) ---
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        trailing_candidate: Optional[str] = None
        for line in reversed(lines):
            if re.search(r'\b\d+\.', line):
                # Once we hit a numbered line we are back in the original PGN context
                break
            if re.match(r'^(plan|analysis|thought|commentary)\b', line, flags=re.I):
                continue
            cleaned_line = re.sub(r'^[A-Za-z ]*[:\-]\s*', '', line)
            tokens = re.findall(SAN_SUB_PATTERN, cleaned_line, flags=re.IGNORECASE)
            if tokens:
                trailing_candidate = tokens[0]
                print(f"<debug> :   trailing_line_candidate: {repr(trailing_candidate)} from line {repr(line)}")
                break
        if trailing_candidate:
            return trailing_candidate

    # Precompute all SAN tokens and their positions
    san_pattern = re.compile(SAN_SUB_PATTERN, flags=re.I)
    san_matches = list(san_pattern.finditer(text))
    san_list = [m.group(0).strip() for m in san_matches]
    san_positions = [m.start() for m in san_matches]

    # Precompute move-number tokens and their positions (e.g. "22.", "22...")
    moveno_pattern = re.compile(r'\b(\d+)\.(?:\.\.)?', flags=re.I)
    moveno_matches = list(moveno_pattern.finditer(text))
    moveno_list = [(int(m.group(1)), m.start(), m.end()) for m in moveno_matches]

    # Helper: find nearest preceding move-number for a given position
    def preceding_moveno(pos):
        prev = None
        for num, s, e in moveno_list:
            if s <= pos:
                prev = (num, s, e)
            else:
                break
        return prev  # (num, start, end) or None

    # --- 3) explicit numbered move for current_turn_number if provided ---
    if current_turn_number is not None:
        turn = int(current_turn_number)
        print(f"<debug> :   looking for explicit move {turn} ({'white' if is_white_to_move else 'black'})")
        # for black: look for "22..." pattern
        if not is_white_to_move:
            m = re.search(rf'\b{turn}\.\.\.\s*({SAN_SUB_PATTERN})', text, flags=re.I)
            if m:
                result = m.group(1).strip()
                print(f"<debug> :   found explicit black move {turn}...: {repr(result)}")
                return result
        else:
            m = re.search(rf'\b{turn}\.\s*({SAN_SUB_PATTERN})', text, flags=re.I)
            if m:
                result = m.group(1).strip()
                print(f"<debug> :   found explicit white move {turn}.: {repr(result)}")
                return result
        print(f"<debug> :   no explicit move found for turn {turn}")

    # --- 4) prefer SANs whose nearest preceding move-number <= current_turn_number ---
    if current_turn_number is not None and san_matches:
        candidates = []
        for san, pos in zip(san_list, san_positions):
            pm = preceding_moveno(pos)
            if pm is None:
                # treat as early in text -> good candidate
                candidates.append((san, pos, -1))
            else:
                pm_num = pm[0]
                candidates.append((san, pos, pm_num))
        # prefer SANs with pm_num <= current_turn_number, and among them choose the rightmost
        valid = [c for c in candidates if c[2] == -1 or c[2] <= current_turn_number]
        if valid:
            # If black to move, prefer the last valid SAN that is plausible
            # If white to move, prefer the first valid SAN (white move usually appears before reply)
            if is_white_to_move:
                return valid[0][0]
            else:
                return valid[-1][0]

    # --- 2) SAN followed by a result token (fallback - only if other rules didn't match) ---
    # This is a fallback for when models don't use proper move numbering
    # Only use if we have a turn number and sequential parsing didn't find a match
    if current_turn_number is None or not sequential_moves:
        san_with_result = re.search(r'(' + SAN_SUB_PATTERN + r')\s+(?:0-1|1-0|1/2-1/2)\b', text, flags=re.I)
        if san_with_result:
            result = san_with_result.group(1).strip()
            print(f"<debug> :   found SAN with result (fallback): {repr(result)}")
            return result

    # --- 5) ellipsis-based explicit black moves anywhere in text ---
    black_pattern = re.compile(r'\b\d+\.\.\.\s*(' + SAN_SUB_PATTERN + r')', flags=re.I)
    black_moves = [m.group(1).strip() for m in black_pattern.finditer(text)]
    if black_moves and not is_white_to_move:
        return black_moves[-1]

    # --- 6) generic explicit white pattern ---
    white_pattern = re.compile(r'\b\d+\.\s*(' + SAN_SUB_PATTERN + r')', flags=re.I)
    white_moves = [m.group(1).strip() for m in white_pattern.finditer(text)]
    if white_moves and is_white_to_move:
        return white_moves[-1]

    # --- 7) Fallback heuristics ---
    if not san_list:
        return None

    # if model is white-to-move pick the first SAN (white tends to appear first)
    if is_white_to_move:
        return san_list[0]

    # if model is black-to-move, use more sophisticated logic
    if not is_white_to_move:
        # For black moves, look for patterns like "Qf3 Bg4" where Qf3 is the black move
        # Try to find alternating patterns: if we have an even number of moves, 
        # black moves are at odd positions (1, 3, 5, ...)
        if len(san_list) >= 2:
            # If we have an even number of moves, black moves are at odd indices
            if len(san_list) % 2 == 0:
                # Return the last odd-indexed move (last black move)
                return san_list[-1]
            else:
                # If odd number of moves, black moves are at even indices
                # Return the last even-indexed move
                return san_list[-2] if len(san_list) >= 2 else san_list[-1]
        else:
            # Single move - if it's black's turn, this should be a black move
            return san_list[0]

    # last resort: return the last SAN
    result = san_list[-1]
    print(f"<debug> :   final_result: {repr(result)}")
    return result


def get_board_from_pgn(pgn_text: str) -> chess.Board:
    """
    Get chess board from PGN text.
    
    Args:
        pgn_text (str): PGN string
        
    Returns:
        chess.Board: Chess board after playing all moves
    """
    pgn_io = StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


def get_fen_from_pgn(pgn_text: str) -> str:
    """
    Get FEN string from PGN text.
    
    Args:
        pgn_text (str): PGN string
        
    Returns:
        str: FEN string of final position
    """
    board = get_board_from_pgn(pgn_text)
    return board.fen()


if __name__ == "__main__":
    # Example usage and tests
    print("Chess Utils Module - Example Usage")
    print("=" * 40)
    
    # Test SAN extraction
    test_cases = [
        ("22... Qxh2+ 23. Kf1 Qh1#", 22, False, "Qxh2+"),
        ("28... Qe1#", 28, False, "Qe1#"),
        ("15. Rxg7+ Kh8 16. Rxf7+ Kg8", 15, True, "Rxg7+"),
        ("Qe1# 0-1 29. Kxe1", 28, False, "Qe1#"),
        # Test cases with newlines interleaved
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6", 3, True, "Bb5"),
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6\n4. Ba4 Nf6", 4, True, "Ba4"),
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6\n4. Ba4 Nf6\n5. O-O Be7", 5, True, "O-O"),
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6\n4. Ba4 Nf6\n5. O-O Be7\n6. Re1 b5", 6, True, "Re1"),

        # Test with mixed whitespace and newlines
        ("1. e4 e5\n\n2. Nf3 Nc6\n\n3. Bb5 a6", 3, True, "Bb5"),
        ("1. e4 e5\n  \n2. Nf3 Nc6\n  \n3. Bb5 a6", 3, True, "Bb5"),
        # Test with explicit move numbers and newlines - should extract correct move for given turn
        ("60. Qg2 Bh3\n61. Qf3 Bg4\n62. Qg2 Bh3", 60, True, "Qg2"),
        ("60. Qg2 Bh3\n61... Qf3\n62. Qg2 Bh3", 61, False, "Qf3"),  # Black move with proper "61..." format
        ("60. Qg2 Bh3\n61. Qf3 Bg4\n62. Qg2 Bh3", 62, True, "Qg2"),  # White move for turn 62
        # Test the specific e4 case mentioned in the user query
        ("1. e4", 1, True, "e4"),
        ("1. e4 e5", 1, True, "e4"),
        ("1. e4 e5\n2. Nf3", 2, True, "Nf3"),
        # Test cases with explicit \n characters
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6", 1, True, "e4"),
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6", 2, True, "Nf3"),
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6", 3, True, "Bb5"),
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6", 1, False, "e5"),
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6", 2, False, "Nc6"),
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6", 3, False, "a6"),
        # Test with longer sequences and explicit \n
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6\n4. Ba4 Nf6\n5. O-O Be7\n6. Re1 b5\n7. Bb3 O-O\n8. c3 d6", 8, True, "c3"),
        ("1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6\n4. Ba4 Nf6\n5. O-O Be7\n6. Re1 b5\n7. Bb3 O-O\n8. c3 d6", 8, False, "d6"),
        # Test with model response format containing \n
        ("Model output: '1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6\n4. Ba4 Nf6\n5. O-O Be7'", 5, True, "O-O"),
        ("Neutral GM response: '1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6\n4. Ba4 Nf6\n5. O-O Be7'", 5, True, "O-O"),
        # Test with checkmate moves and \n
        ("48. Rh8+ Rg8\n49. Rxg8#", 48, True, "Rh8+"),
        ("48. Rh8+ Rg8\n49. Rxg8#", 49, True, "Rxg8#"),
        # Test with complex moves and \n
        ("25. Nh4 b3\n26. a3 Bb5\n27. Qf3 Bd7", 25, True, "Nh4"),
        ("25. Nh4 b3\n26. a3 Bb5\n27. Qf3 Bd7", 26, True, "a3"),
        ("25. Nh4 b3\n26. a3 Bb5\n27. Qf3 Bd7", 27, True, "Qf3"),
        # Test the long repetitive game sequence - should extract e4 for move 1 white
        ("'1. e4 e5\n2. Nf3 Nc6\n3. Bb5 a6\n4. Ba4 Nf6\n5. O-O Be7\n6. Re1 b5\n7. Bb3 O-O\n8. c3 d5\n9. exd5 Nxd5\n10. Nxe5 Nxe5\n11. Rxe5 c6\n12. d4 Bd6\n13. Re1 Qh4\n14. g3 Qh3\n15. Be3 Bg4\n16. Qd3 Rae8\n17. Nd2 Re6\n18. Qf1 Qh5\n19. a4 Rfe8\n20. axb5 axb5\n21. Bxd5 Qxd5\n22. Qg2 Qh5\n23. Ra6 Bf8\n24. Rxc6 Bh3\n25. Qf3 Bg4\n26. Qg2 Bh3\n27. Qf3 Bg4\n28. Qg2 Bh3\n29. Qf3 Bg4\n30. Qg2 Bh3\n31. Qf3 Bg4\n32. Qg2 Bh3\n33. Qf3 Bg4\n34. Qg2 Bh3\n35. Qf3 Bg4\n36. Qg2 Bh3\n37. Qf3 Bg4\n38. Qg2 Bh3\n39. Qf3 Bg4\n40. Qg2 Bh3\n41. Qf3 Bg4\n42. Qg2 Bh3\n43. Qf3 Bg4\n44. Qg2 Bh3\n45. Qf3 Bg4\n46. Qg2 Bh3\n47. Qf3 Bg4\n48. Qg2 Bh3\n49. Qf3 Bg4\n50. Qg2 Bh3\n51. Qf3 Bg4\n52. Qg2 Bh3\n53. Qf3 Bg4\n54. Qg2 Bh3\n55. Qf3 Bg4\n56. Qg2 Bh3\n57. Qf3 Bg4\n58. Qg2 Bh3\n59. Qf3 Bg4\n60. Qg2 Bh3'", 1, True, "e4"),
    ]
    
    for text, turn_num, is_white, expected in test_cases:
        result = extract_predicted_move(text, current_turn_number=turn_num, is_white_to_move=is_white)
        print(f"Input: {text}")
        print(f"Expected: {expected}, Got: {result}")
        print("---")

