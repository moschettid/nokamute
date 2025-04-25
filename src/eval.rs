use crate::board::*;
use crate::bug::Bug;
use crate::hex_grid::*;

use minimax::{Evaluation, Evaluator};

// An evaluator that knows nothing but the rules, and maximally explores the tree.
pub struct DumbEvaluator;

impl Evaluator for DumbEvaluator {
    type G = Rules;
    fn evaluate(&self, _: &Board) -> Evaluation {
        0
    }
}

// An evaluator that counts movable pieces and how close to death the queen is.
#[derive(Copy, Clone)]
pub struct BasicEvaluator {
    aggression: Evaluation,
    queen_liberty_factor: Evaluation,
    gates_factor: Evaluation,
    queen_spawn_factor: Evaluation,
    movable_queen_value: Evaluation,
    movable_bug_factor: Evaluation,
    unplayed_bug_factor: Evaluation,
    // Bonus for defensive pillbug or placeability thereof.
    pillbug_defense_bonus: Evaluation,
}

// Ideas:
//  - High level aggression setting
//    - Value mobility higher
//    - Opponent's mobility is more negative than you're mobility is positive.
//    - Don't value queen factor highly until a large mobility advantage is established.
//    - Quadratic score for filling queen liberties (counting virtual pillbug liberties)
//    - Conservative option: ignore queen and try to shut out opponent first.
//        Need to count placeable positions
//  - Directly encode "qualify for win" separate from queen liberties.
//    - Is the queen next to pillbug powers with an escape route?
//    - Is there a placeable position next to queen with a pillbug available?

impl BasicEvaluator {
    pub(crate) fn new(aggression: u8) -> Self {
        // Ensure aggression is a dial between 1 and 5.
        let aggression = aggression.clamp(1, 5) as Evaluation;
        Self {
            aggression,
            queen_liberty_factor: 50, //aggression * 10,
            ///////////////////////////////////////////
            //v0.1.1, v0.1.1.1 with check on grasshopper and ladybug
            gates_factor: (6-aggression) * 10,//NEED TO FIND THE RIGHT VALUE, value winning against nokamute: 4
            ///////////////////////////////////////////
            //v0.1.3.. not good
            queen_spawn_factor: aggression * 8,//aggression * 2
            ///////////////////////////////////////////
            movable_queen_value: 1,//aggression*4
            movable_bug_factor: 2,//2
            ///////////////////////////////////////////
            //v0.1.2 we have added the spawn flag for unplayed bugs
            unplayed_bug_factor: 1,//1
            ///////////////////////////////////////////
            pillbug_defense_bonus: (6-aggression) * 40,//aggression * 40
        }
    }

    pub(crate) fn aggression(&self) -> u8 {
        self.aggression as u8
    }

    fn value(&self, bug: Bug) -> Evaluation {
        // Mostly made up. All I know is that ants are good.
        match bug {
            Bug::Queen => 3,//self.movable_queen_value
            Bug::Ant => 7,
            Bug::Beetle => 6,
            Bug::Grasshopper => 3,//2, 4 with the strong gate check
            Bug::Spider => 2,
            Bug::Mosquito => 8, // See below.
            Bug::Ladybug => 4, //6
            Bug::Pillbug => 5,
        }
    }
}

impl Default for BasicEvaluator {
    fn default() -> Self {
        Self::new(3)
    }
}

//COMMENT: adj == origin not needed for this function..
//ape regina deve scappare non è una liberty la posizione di partenza
fn count_liberties(board: &Board, origin: Hex, hex: Hex) -> Evaluation {
    adjacent(hex).into_iter().filter(|&adj| adj == origin || !board.occupied(adj)).count()
        as Evaluation
}

fn placeable(board: &Board, hex: Hex, color: Color) -> bool {
    !adjacent(hex).iter().any(|&adj| board.occupied(adj) && board.node(adj).color() != color)
}

#[test]
fn test_placeable() {
    let b = Board::from_game_string("Base;;;wA1;bA1 wA1-;wA2 /wA1").unwrap();
    assert!(!placeable(&b, Direction::SE.apply(START_HEX), Color::White));
    assert!(!placeable(&b, Direction::NE.apply(START_HEX), Color::White));
    assert!(placeable(&b, Direction::NW.apply(START_HEX), Color::White));
    assert!(!placeable(&b, Direction::SE.apply(START_HEX), Color::Black));
    assert!(!placeable(&b, Direction::NE.apply(START_HEX), Color::Black));
    assert!(!placeable(&b, Direction::NW.apply(START_HEX), Color::Black));
}

impl Evaluator for BasicEvaluator {
    type G = Rules;

    fn evaluate(&self, board: &Board) -> Evaluation {
        let mut buf = [0; 6];
        let mut immovable = board.find_cut_vertexes();

        let mut score = 0;
        let mut pillbug_defense = [false; 2];
        let mut queen_score = [0; 2];
        let mut gates_score = [0; 2];
        let mut unplayed_bug_score = 0;

        // Check for spawn points.
        let spawn_flag = spawn_points_flag(board, board.to_move() as usize);
        let spawn_flag_opponent = spawn_points_flag(board, board.to_move().other() as usize);
        let remaining = board.get_remaining();
        let opp_remaining = board.get_opponent_remaining();
        for bug in Bug::iter_all() {
            if spawn_flag && remaining[bug as usize] > 0 {
                unplayed_bug_score += (remaining[bug as usize] as Evaluation)
                    * self.unplayed_bug_factor
                    * self.value(bug);
            }
            if spawn_flag_opponent && opp_remaining[bug as usize] > 0 {
                unplayed_bug_score -= (opp_remaining[bug as usize] as Evaluation)
                    * self.unplayed_bug_factor
                    * self.value(bug);
            }
        }

        for &hex in board.occupied_hexes[0].iter().chain(board.occupied_hexes[1].iter()) {
            let node = board.node(hex);
            let mut bug_score = self.value(node.bug());
            let mut pillbug_powers = node.bug() == Bug::Pillbug;
            let mut crawler = node.bug().crawler();
            if node.bug() == Bug::Mosquito {
                // Mosquitos are valued as they can currently move.
                bug_score = 0;
                crawler = true;
                if node.is_stacked() {
                    bug_score = self.value(Bug::Beetle);
                } else {
                    for adj in adjacent(hex) {
                        if board.occupied(adj) {
                            let bug = board.node(adj).bug();
                            if bug != Bug::Queen {
                                bug_score = self.value(bug);
                            }
                            if bug == Bug::Pillbug {
                                pillbug_powers = true;
                            }
                            if !bug.crawler() {
                                crawler = false;
                            }
                        }
                    }
                }
            };
            //errore, qui il mosquito prende lo score dall'ultimo degli adj!

            if crawler {
                // Treat blocked crawlers as immovable.
                if board.slidable_adjacent(&mut buf, hex, hex).next().is_none() {
                    immovable.set(hex);
                }
            }

            if node.is_stacked() {
                bug_score *= 2;
            }

            let friendly_queen = board.queens[node.color() as usize];

            // TODO: transpose this loop, i.e. categorize queen liberties after the bug loop.
            // Count libs for more if they are not crawlable (e.g. behind a gate)
            if adjacent(friendly_queen).contains(&hex) {
                // Filling friendly queen's liberty.
                if immovable.get(hex) && !node.is_stacked() {
                    queen_score[node.color() as usize] -= self.queen_liberty_factor;
                } else {
                    // Lower penalty for being able to leave.
                    queen_score[node.color() as usize] -= self.queen_liberty_factor / 3;//original: /2
                }
                if pillbug_powers && board.node(friendly_queen).clipped_height() == 1 {
                    let best_escape = adjacent(hex)
                        .into_iter()
                        .map(|lib| {
                            if board.occupied(lib) {
                                0
                            } else {
                                count_liberties(board, friendly_queen, lib)
                            }
                        })
                        .max()
                        .unwrap_or(0);
                    // maybe also best escape == 1 or 2 can be a good idea
                    if best_escape > 1 {//original: > 2
                        pillbug_defense[node.color() as usize] = true;
                    }
                }
            }

            let enemy_queen = board.queens[node.color().other()];

            if adjacent(enemy_queen).contains(&hex) {
                // Discourage liberty filling by valuable bugs, by setting their score to zero when filling a liberty.
                bug_score = 0;
                // A little extra boost for filling opponent's queen, as we will never choose to move.
                queen_score[node.color().other()] -= self.queen_liberty_factor * 2; //original: * 12 / 10
                if pillbug_powers {
                    let best_unescape = adjacent(hex)
                        .into_iter()
                        .map(|lib| {
                            if board.occupied(lib) {
                                6
                            } else {
                                count_liberties(board, enemy_queen, lib)
                            }
                        })
                        .min()
                        .unwrap_or(6);
                    if best_unescape < 4 { //original: < 3
                        queen_score[node.color().other()] = -self.queen_liberty_factor;
                    }
                }
            }

            if !node.is_stacked() && immovable.get(hex) {
                // Pinned bugs are worthless.
                continue;
            }

            bug_score *= self.movable_bug_factor;
            if node.color() != board.to_move() {
                bug_score = -bug_score;
                // Make low-aggression mode value opponent movability higher than ours.
                if self.aggression == 1 {
                    bug_score *= 2
                } else if self.aggression == 2 {
                    bug_score = bug_score * 3 / 2;
                }
            }
            score += bug_score;
        }

        let mut pillbug_defense_score = self.pillbug_defense_bonus
            * (pillbug_defense[board.to_move() as usize] as Evaluation
                - pillbug_defense[board.to_move().other()] as Evaluation);

        // Check for backup defensive pillbug placeability option, discounted value
        pillbug_defense = [false; 2];
        for &color in &[Color::Black, Color::White] {
            if board.node(board.queens[color as usize]).clipped_height() == 1
                && board.remaining[color as usize][Bug::Pillbug as usize] > 0
                && adjacent(board.queens[color as usize])
                    .iter()
                    .any(|&lib| placeable(board, lib, color))
            {
                pillbug_defense[color as usize] = true;
            }
        }
        pillbug_defense_score += self.pillbug_defense_bonus / 2
            * (pillbug_defense[board.to_move() as usize] as Evaluation
                - pillbug_defense[board.to_move().other()] as Evaluation);

        
        // Check for gates.
        //try to do a more spefic check: check if there ara grasshopper that can jump in or there are free grasshopper
        gates_score[board.to_move() as usize] += self.gates_factor*check_gates(board, board.to_move() as usize)*(4-count_free_grasshoppers(board, board.to_move().other(), immovable))/4;
        gates_score[board.to_move().other()] -= self.gates_factor*check_gates(board, board.to_move().other())*(4-count_free_grasshoppers(board, board.to_move() as usize, immovable))/4;
        let gates_score = gates_score[board.to_move() as usize] - gates_score[board.to_move().other()];
        
        // Check for spawn points next to opponent queen
        // before check if there is an available beetle, otherwise the spawn points will be 0
        //check also if there are bugs that can easily pin our beetle in the future
        let mut queen_spawn_score = 0;
        if remaining_available_beetle(board, board.to_move() as usize){
            if pinning_beatle_pieces(board, board.to_move().other() as usize){
                queen_spawn_score = self.queen_spawn_factor*count_queen_spawn_points(board, board.to_move() as usize)/16;
            }else{
                queen_spawn_score = self.queen_spawn_factor*count_queen_spawn_points(board, board.to_move() as usize);
            }            
        }
        let mut queen_spawn_score_opponent = 0;
        if remaining_available_beetle(board, board.to_move().other() as usize) && !pinning_beatle_pieces(board, board.to_move() as usize){
            if pinning_beatle_pieces(board, board.to_move().other() as usize){
                queen_spawn_score_opponent = self.queen_spawn_factor*count_queen_spawn_points(board, board.to_move().other() as usize)/16;
            }else{
                queen_spawn_score_opponent = self.queen_spawn_factor*count_queen_spawn_points(board, board.to_move().other() as usize);
            }
        }
        
        let queen_spawn_score = queen_spawn_score - queen_spawn_score_opponent;
        
        
        let queen_score = queen_score[board.to_move() as usize] - queen_score[board.to_move().other()];
        queen_score + pillbug_defense_score + score + gates_score + unplayed_bug_score + queen_spawn_score
    }

    // The idea here is to use quiescence search to avoid ending on a
    // placement. This is based on the hypothesis that new pieces are placed
    // with the intention of moving them on the next turn. Stopping the search
    // just after placing a piece can give bad results because it would
    // usually pins one of your own pieces and doesn't put the new piece where
    // it will be useful. Thus, each player can get a bonus move to move a
    // piece that they have just placed (but not other pieces).
    fn generate_noisy_moves(&self, board: &Board, moves: &mut Vec<Turn>) {
        if board.turn_history.len() < 4 || board.get_remaining()[Bug::Queen as usize] == 0 {
            // Wait until movements are at least possible.
            return;
        }
        let enemy_last_move = board.turn_history[board.turn_history.len() - 1];
        let my_last_move = board.turn_history[board.turn_history.len() - 2];

        if let Turn::Place(hex, _) = my_last_move {
            // Drop attack is quiet enough.
            if !adjacent(board.queens[board.to_move().other()]).contains(&hex) {
                // TODO: just generate from this spot (ignoring throws?).
                board.generate_movements(moves);
                moves.retain(|m| if let Turn::Move(start, _) = *m { start == hex } else { false });
                // If the piece became pinned or covered, this will return no
                // moves, which means the search will terminate.
                return;
            }
        }

        if let Turn::Place(hex, _) = enemy_last_move {
            if !adjacent(board.queens[board.to_move() as usize]).contains(&hex) {
                // We didn't just place, but opponent did. Do some movement to
                // give them a chance to quiesce.
                board.generate_movements(moves);
            }
        }

        // If no one just placed something, return nothing and stop the search.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimax() {
        use minimax::{Negamax, Strategy};

        // Find the winning move.
        // ．．．🐝🕷．．
        //．．🐜🐜🐝．．
        // ．．．🦗🪲
        let mut board = Board::default();
        board.apply(Turn::Place(loc_to_hex((0, 0)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((1, 0)), Bug::Spider));
        board.apply(Turn::Place(loc_to_hex((-1, 1)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((0, 1)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((1, 2)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((1, 1)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((2, 2)), Bug::Beetle));
        board.apply(Turn::Pass);
        for depth in 1..3 {
            let mut strategy = Negamax::new(DumbEvaluator {}, depth);
            let m = strategy.choose_move(&mut board);
            assert_eq!(Some(Turn::Move(loc_to_hex((-1, 1)), loc_to_hex((2, 1)))), m);

            let mut strategy = Negamax::new(BasicEvaluator::default(), depth);
            let m = strategy.choose_move(&mut board);
            assert_eq!(Some(Turn::Move(loc_to_hex((-1, 1)), loc_to_hex((2, 1)))), m);
        }

        // Find queen escape.
        //．．🕷🐝🐝．
        // ．．🦗🕷．
        let mut board = Board::default();
        board.apply(Turn::Place(loc_to_hex((0, 0)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((1, 0)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((1, 1)), Bug::Spider));
        board.apply(Turn::Place(loc_to_hex((0, 1)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((-1, 0)), Bug::Beetle));
        board.apply(Turn::Pass);
        for depth in 1..3 {
            let mut strategy = Negamax::new(BasicEvaluator::default(), depth);
            let m = strategy.choose_move(&mut board);
            assert_eq!(Some(Turn::Move(loc_to_hex((0, 0)), loc_to_hex((0, -1)))), m);
        }
    }
}


// TODO: check about the Qualify to win condition for a given player and set consequently the aggression value
// This condition checks if: there is no opponent pillbug near the opponet queen 
// if there is one, check if our beetle are on the opponent queen or on the opponent pillbug
// or if the opponent pillbug has no liberties to move the queen

/*
fn qualify_to_win(board: &Board, color: Color) -> bool {
    let queen = board.queens[color as usize];
    let pillbug = board.node(queen).bug() == Bug::Pillbug;
    let mut beetle = false;
    for adj in adjacent(queen) {
        if board.node(adj).bug() == Bug::Pillbug {
            return false;
        }
        if board.node(adj).bug() == Bug::Beetle {
            beetle = true;
        }
    }
    if !beetle && pillbug {
        return false;
    }
    true
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////

fn is_there_a_filling_grasshopper(board: &Board, color: usize, hex: Hex) -> bool {
    for &hexy in board.occupied_hexes[color as usize].iter() {
        let node = board.node(hexy);
        if node.bug() == Bug::Grasshopper {
            // Check if the grasshopper can go in Hex with a move
            // check if hex is in the hexes reachable by the grasshopper
            for dir in Direction::all() {
                let mut jump = dir.apply(hexy);
                let mut dist = 1;
                while board.occupied(jump) {
                    jump = dir.apply(jump);
                    dist += 1;
                    if jump == hexy {
                        // Exit out if we'd infinitey loop.
                        dist = 0;
                        break;
                    }
                }
                if dist > 1 {
                    if jump == hex {
                        // Check if the grasshopper can jump to the hex
                        return true; // Found a filling grasshopper
                    }
                }
            }
        }
    }
    return false // No filling grasshopper found
}

fn is_there_a_filling_ladybug(board: &Board, color: usize, hex: Hex) -> bool {
    for &hexy in board.occupied_hexes[color as usize].iter() {
        let node = board.node(hexy);
        if node.bug() == Bug::Ladybug {
            // Check if the ladybug can go in Hex with a move
            // check if hex is in the hexes reachable by the ladybug
            let mut buf1 = [0; 6];
            let mut buf2 = [0; 6];
            let mut buf3 = [0; 6];
            let mut step2 = HexSet::new();
            let mut step3 = HexSet::new();
            for s1 in board.slidable_adjacent_beetle(&mut buf1, hexy, hexy) {
                if board.occupied(s1) {
                    for s2 in board.slidable_adjacent_beetle(&mut buf2, hexy, s1) {
                        if board.occupied(s2) && !step2.get(s2) {
                            step2.set(s2);
                            for s3 in board.slidable_adjacent_beetle(&mut buf3, hexy, s2) {
                                if !board.occupied(s3) && !step3.get(s3) {
                                    step3.set(s3);
                                    if s3 == hex {
                                        // Check if the ladybug can jump to the hex
                                        return true; // Found a filling ladybug
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return false // No filling ladybug found
}

fn count_free_grasshoppers(board: &Board, color: usize, immovable: &HexSet) -> Evaluation {
    let mut count = 0;

    // Count grasshoppers already on the board and free to move
    for &hex in board.occupied_hexes[color as usize].iter() {
        let node = board.node(hex);
        if node.bug() == Bug::Grasshopper {
            // Check if the grasshopper is immovable, otherwise count it
            if !immovable.get(hex) {
                count++; // Skip immovable grasshoppers
            }
        }
    }

    // Add the number of unplayed grasshoppers
    count += board.remaining[color as usize][Bug::Grasshopper as usize] as Evaluation;

    count // Return the total count of free grasshoppers
}

fn check_gates(board: &Board, color: usize) -> Evaluation { //Hex
    let queen = board.queens[color as usize];
    let mut gates = 0;
    //let mut gate_hex = loc_to_hex((0, 0));
    for adj in adjacent(queen) {
        //check if adj is occupied, if so go to the next itaration
        if board.occupied(adj) {
            continue;
        }
        if is_gate(board, adj) {
            //check if there is a filling grasshopper
            if is_there_a_filling_grasshopper(board, 1-color, adj) || is_there_a_filling_ladybug(board, 1-color, adj) {
                continue;
            }
            //gate_hex = adj;
            gates += 1;
        }
    }
    gates
    //gate_hex
}
// define a function that checks if an hex is a gate
//idea: check if a crawler can move to the hex using slide_board.slidable_adjacent
//if not, return true
fn is_gate(board: &Board, hex: Hex) -> bool {
    //assumptions: the hex is free
    let mut neighbors = [0; 6]; // Mutable array to hold neighbors.
    let origin = hex; // Use the same hex as origin if no specific origin is needed.
    return board.slidable_adjacent(&mut neighbors, origin, hex).collect().is_empty(); // Otherwise, it's not a gate.
}

////////////////////////////////////////////////////////////////////////////////////////////////

fn spawn_points_flag(board: &Board, color: usize) -> bool {
    for &hex in board.occupied_hexes[color as usize].iter() {
        for adj in adjacent(hex) {
            // Check if the adjacent hex is unoccupied and placeable for the given color
            if !board.occupied(adj) && placeable(board, adj, Color::from_usize(color)) {
                return true; // Found a spawn point
            }
        }
    }
    return false; // No spawn points found
}

////////////////////////////////////////////////////////////////////////////////////////////////

//function to see if there is an available beetle among unplaced pieces
fn remaining_available_beetle(board: &Board, color: usize) -> bool {
    //check if there is an available beetle among unplaced pieces
    return board.remaining[color as usize][Bug::Beetle as usize] > 0;
}

//aggiungi il colore
fn pinning_beatle_pieces(board: &Board, color: usize, immovable: &HexSet) -> bool {
    //check if there are bugs that can easily pin our beetle in the future
    for &hex in board.occupied_hexes[color as usize].iter() {
        let node = board.node(hex);
        //check if there are movable Ants, Mosquitos next to Ants
        if node.bug() == Bug::Ant {
            //check if the node can move
            if !immovable.get(hex) {
                return true; // Skip immovable grasshoppers
            }
        }
        if node.bug() == Bug::Mosquito {
            //check if the node can move
            //check if an adjacent is occupied by an Ant
            for adj in adjacent(hex) {
                if board.occupied(adj) && board.node(adj).bug() == Bug::Ant {
                    if !immovable.get(hex) {
                        count++; // Skip immovable grasshoppers
                    }
                }
            }
        }
    }
    false // No pinned beetle found
    //NOTE: here Spider, Ladybug, Grasshopper are not considered as they are more unlikely to pin a beetle
}

//function to count the spawn points near a peice next to the adversary queen
fn count_queen_spawn_points(board: &Board, color: usize) -> Evaluation {
    //opponent queen
    let opp_queen = board.queens[1-(color as usize)];//not opponent queen
    let mut queen_spawn = 0;
    //check if there are our pieces next to the opponent queen
    for adj in adjacent(opp_queen) {
        //check if the hex is occupied by our pieces
        if board.occupied(adj) && board.node(adj).color() == Color::from_usize(color) {
            //now check if an adjacent hex to this adj is unoccupied and placeable for us
            for space in adjacent(adj) {
                //check if the hex is unoccupied and placeable for us
                if !board.occupied(space) && placeable(board, space, Color::from_usize(color)) {
                    queen_spawn += 1;
                }
            }
        }
    }
    queen_spawn
}

////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test_gating {
    use super::*;
    #[test]
    fn test_gates() {
        let mut board = Board::default();
        board.apply(Turn::Place(loc_to_hex((0, 0)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((1, 0)), Bug::Spider));
        board.apply(Turn::Place(loc_to_hex((-1, 1)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((2, 3)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((1, 2)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((0, 1)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((2, 2)), Bug::Beetle));
        board.apply(Turn::Pass);
        assert_eq!(check_gates(&board, 1), 1);
        assert_eq!(is_there_a_filling_grasshopper(&board, 0, loc_to_hex((1, 1))), false);

        let mut board = Board::default();
        board.apply(Turn::Place(loc_to_hex((0, 0)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((1, 0)), Bug::Spider));
        board.apply(Turn::Place(loc_to_hex((-1, 1)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((2, 3)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((1, 2)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((0, 1)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((2, 2)), Bug::Beetle));
        board.apply(Turn::Pass);
        assert_eq!(is_there_a_filling_grasshopper(&board, 0, loc_to_hex((1, 1))), true);
        assert_eq!(check_gates(&board, 1), 0);

        let mut board = Board::default();
        board.apply(Turn::Place(loc_to_hex((0, 0)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((1, 0)), Bug::Spider));
        board.apply(Turn::Place(loc_to_hex((-1, 1)), Bug::Ladybug));
        board.apply(Turn::Place(loc_to_hex((2, 3)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((1, 2)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((0, 1)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((2, 2)), Bug::Beetle));
        board.apply(Turn::Pass);
        assert_eq!(is_there_a_filling_ladybug(&board, 0, loc_to_hex((1, 1))), true);
        assert_eq!(check_gates(&board, 1), 0);
    }
}
//PASSED!

// TODO: implement a test for the spawn_points_flag function. This is implemented by copilot
#[cfg(test)]
mod test_spawn_points_flag {
    use super::*;
    #[test]
    fn test_spawn_points_flag() {
        let mut board = Board::default();
        board.apply(Turn::Place(loc_to_hex((0, 0)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((1, 0)), Bug::Spider));
        board.apply(Turn::Place(loc_to_hex((-1, 1)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((2, 3)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((1, 2)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((0, 1)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((2, 2)), Bug::Beetle));
        board.apply(Turn::Pass);
        assert_eq!(spawn_points_flag(&board, 1), true);
    }
}

#[cfg(test)]
mod test_count_queen_spawn_points {
    use super::*;
    #[test]
    fn test_count_queen_spawn_points() {
        let mut board = Board::default();
        board.apply(Turn::Place(loc_to_hex((0, 0)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((1, 0)), Bug::Spider));
        board.apply(Turn::Place(loc_to_hex((-1, 1)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((2, 3)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((1, 2)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((0, 1)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((2, 2)), Bug::Beetle));
        board.apply(Turn::Pass);
        assert_eq!(count_queen_spawn_points(&board, 0), 4);
        assert_eq!(count_queen_spawn_points(&board, 1), 2);
        assert_eq!(pinning_beatle_pieces(&board, 1), true);

        let mut board = Board::default();
        board.apply(Turn::Place(loc_to_hex((0, 0)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((1, 0)), Bug::Spider));
        board.apply(Turn::Place(loc_to_hex((-1, 1)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((2, 3)), Bug::Spider));
        board.apply(Turn::Place(loc_to_hex((1, 2)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((0, 1)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((2, 2)), Bug::Beetle));
        board.apply(Turn::Pass);
        assert_eq!(count_queen_spawn_points(&board, 0), 4);
        assert_eq!(count_queen_spawn_points(&board, 1), 2);
        assert_eq!(pinning_beatle_pieces(&board, 1), false);
    }
}


/*
TODO:
- Spawn points near the queen, need improvements
- Ant games
- Fixed starting moves
- qualify to win condition
- manage better the aggression value
*/

/*
Added from nokamute:
- Check for gates
- Flag for spawn points for unplayed bugs (not perfect)
- Check for spawn points near the queen
*/

//Last thing done: spawn points near the queen

//errore riga 173