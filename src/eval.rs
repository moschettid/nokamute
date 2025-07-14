use crate::board::*;
use crate::bug::Bug;
use crate::hex_grid::*;

use minimax::{Evaluation, Evaluator};

// An evaluator that knows nothing but the rules, and maximally explores the tree.
pub struct DumbEvaluator;

// TO DO:
// - clone vanilla nokamute and import our modification in a parametrized/annullable way
// - decimal values for params
// - add new game mechanincs to the evaluation
// - REFACTOR

impl Evaluator for DumbEvaluator {
    type G = Rules;
    fn evaluate(&self, _: &Board) -> Evaluation {
        0
    }
}

// An evaluator that uses come basic heuristics to evaluate the board state.
#[derive(Copy, Clone)]
pub struct BasicEvaluator {
    queen_liberty_penalty: f32,
    gates_factor: f32,
    queen_spawn_factor: f32,
    unplayed_bug_factor: f32,
    pillbug_defense_bonus: f32,
    ant_game_factor: f32,
    queen_score: f32,
    ant_score: f32,
    beetle_score: f32,
    grasshopper_score: f32,
    spider_score: f32,
    mosquito_score: f32,
    ladybug_score: f32,
    pillbug_score: f32,
    mosquito_incremental_score: f32,
    stacked_bug_factor: f32,
    queen_movable_penalty_factor: f32,
    opponent_queen_liberty_penalty_factor: f32,
    trap_queen_penalty: f32,
    placeable_pillbug_defense_bonus: f32,
    pinnable_beetle_factor: f32,
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
    pub(crate) fn new() -> Self {
        Self {
            queen_liberty_penalty: 200.0,
            gates_factor: 10.0,
            queen_spawn_factor: 40.0,
            unplayed_bug_factor: 1.0,
            pillbug_defense_bonus: 40.0,
            ant_game_factor: 25.0,
            queen_score: 3.0,
            ant_score: 7.0,
            beetle_score: 6.0,
            grasshopper_score: 3.0,
            spider_score: 2.0,
            mosquito_score: 8.0,
            ladybug_score: 4.0,
            pillbug_score: 5.0,
            mosquito_incremental_score: 2.0,
            stacked_bug_factor: 4.0,
            queen_movable_penalty_factor: 2.0,
            opponent_queen_liberty_penalty_factor: 2.0,
            trap_queen_penalty: 200.0,
            placeable_pillbug_defense_bonus: 20.0,
            pinnable_beetle_factor: 8.0,
        }
    }

    // Setter methods for each attribute
    pub fn queen_liberty_penalty(&mut self, value: f32) -> &mut Self {
        self.queen_liberty_penalty = value;
        self
    }

    pub fn gates_factor(&mut self, value: f32) -> &mut Self {
        self.gates_factor = value;
        self
    }

    pub fn queen_spawn_factor(&mut self, value: f32) -> &mut Self {
        self.queen_spawn_factor = value;
        self
    }

    pub fn unplayed_bug_factor(&mut self, value: f32) -> &mut Self {
        self.unplayed_bug_factor = value;
        self
    }

    pub fn pillbug_defense_bonus(&mut self, value: f32) -> &mut Self {
        self.pillbug_defense_bonus = value;
        self
    }

    pub fn ant_game_factor(&mut self, value: f32) -> &mut Self {
        self.ant_game_factor = value;
        self
    }

    pub fn queen_score(&mut self, value: f32) -> &mut Self {
        self.queen_score = value;
        self
    }

    pub fn ant_score(&mut self, value: f32) -> &mut Self {
        self.ant_score = value;
        self
    }

    pub fn beetle_score(&mut self, value: f32) -> &mut Self {
        self.beetle_score = value;
        self
    }

    pub fn grasshopper_score(&mut self, value: f32) -> &mut Self {
        self.grasshopper_score = value;
        self
    }

    pub fn spider_score(&mut self, value: f32) -> &mut Self {
        self.spider_score = value;
        self
    }

    pub fn mosquito_score(&mut self, value: f32) -> &mut Self {
        self.mosquito_score = value;
        self
    }

    pub fn ladybug_score(&mut self, value: f32) -> &mut Self {
        self.ladybug_score = value;
        self
    }

    pub fn pillbug_score(&mut self, value: f32) -> &mut Self {
        self.pillbug_score = value;
        self
    }

    pub fn mosquito_incremental_score(&mut self, value: f32) -> &mut Self {
        self.mosquito_incremental_score = value;
        self
    }

    pub fn stacked_bug_factor(&mut self, value: f32) -> &mut Self {
        self.stacked_bug_factor = value;
        self
    }

    pub fn queen_movable_penalty_factor(&mut self, value: f32) -> &mut Self {
        self.queen_movable_penalty_factor = value;
        self
    }

    pub fn opponent_queen_liberty_penalty_factor(&mut self, value: f32) -> &mut Self {
        self.opponent_queen_liberty_penalty_factor = value;
        self
    }

    pub fn trap_queen_penalty(&mut self, value: f32) -> &mut Self {
        self.trap_queen_penalty = value;
        self
    }

    pub fn placeable_pillbug_defense_bonus(&mut self, value: f32) -> &mut Self {
        self.placeable_pillbug_defense_bonus = value;
        self
    }

    pub fn pinnable_beetle_factor(&mut self, value: f32) -> &mut Self {
        self.pinnable_beetle_factor = value;
        self
    }

    fn value(&self, bug: Bug) -> f32 {
        // Mostly made up. All I know is that ants are good.
        match bug {
            Bug::Queen => self.queen_score,
            Bug::Ant => self.ant_score,
            Bug::Beetle => self.beetle_score,
            Bug::Grasshopper => self.grasshopper_score,
            Bug::Spider => self.spider_score,
            Bug::Mosquito => self.mosquito_score,
            Bug::Ladybug => self.ladybug_score,
            Bug::Pillbug => self.pillbug_score,
        }
    }
}

impl Default for BasicEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

fn count_liberties(board: &Board, origin: Hex, hex: Hex) -> u8 {
    adjacent(hex).into_iter().filter(|&adj| adj == origin || !board.occupied(adj)).count() as u8
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
        let pinned = immovable.clone();

        let mut score = 0.0;
        let mut pillbug_defense = [false; 2];
        let mut queen_score = [0.0; 2];
        let mut gates_score = [0.0; 2];
        let mut unplayed_bug_score = 0.0;
        let mut can_pin_beetle = [false; 2];

        // Check for spawn points.
        let mut spawn_flag = spawn_points_flag(board, board.to_move());
        let mut spawn_flag_opponent = spawn_points_flag(board, board.to_move().other());
        let spawn_flag_active = true; // Should paramentrize this TODO
        if !spawn_flag_active {
            //Is the spawn flag strategy not activated, we short-circuit the control
            spawn_flag = true;
            spawn_flag_opponent = true;
        } //Ideally, should be ok to keep it active, since if there are no spawn points, the unplayed bugs are useless
        let remaining = board.get_remaining();
        let opp_remaining = board.get_opponent_remaining();
        for bug in Bug::iter_all() {
            //Slightly modified in order to get more similar to the original version: we count each unplayed bug not only once
            if spawn_flag {
                unplayed_bug_score +=
                    (remaining[bug as usize] as f32) * self.unplayed_bug_factor * self.value(bug);
            }
            if spawn_flag_opponent {
                unplayed_bug_score -= (opp_remaining[bug as usize] as f32)
                    * self.unplayed_bug_factor
                    * self.value(bug);
            }
        }

        // Calculate the difference between the two players' played ants.
        let mut ant = 0;
        let mut ant_opponent = 0;

        for &hex in board.occupied_hexes[0].iter().chain(board.occupied_hexes[1].iter()) {
            let node = board.node(hex);
            let mut bug_score = self.value(node.bug());
            let mut pillbug_powers = node.bug() == Bug::Pillbug;
            let mut crawler = node.bug().crawler();
            let mut mosquito_ant = false;
            if node.bug() == Bug::Mosquito {
                // Mosquitos are valued as they can currently move.
                bug_score = 0.0;
                crawler = true;
                let can_slide_mosquito =
                    board.slidable_adjacent(&mut buf, hex, hex).next().is_some();
                if node.is_stacked() {
                    bug_score = self.value(Bug::Beetle);
                } else {
                    let mut adj_mosquito = false;
                    let mut real_adj = 0;
                    for adj in adjacent(hex) {
                        if board.occupied(adj) {
                            let bug = board.node(adj).bug();
                            real_adj += 1;
                            if bug == Bug::Mosquito {
                                adj_mosquito = true;
                            }
                            if bug != Bug::Queen {
                                if bug.crawler() && bug != Bug::Pillbug && !can_slide_mosquito {
                                    continue;
                                }
                                bug_score = bug_score.max(self.value(bug));
                            }
                            if bug == Bug::Pillbug {
                                pillbug_powers = true;
                            }
                            if !bug.crawler() {
                                crawler = false;
                            }
                            if bug == Bug::Ant {
                                mosquito_ant = true;
                            }
                        }
                    }
                    if adj_mosquito && real_adj == 1 {
                        //In this case, it is immovable
                        immovable.set(hex);
                        continue;
                    }
                    bug_score += self.mosquito_incremental_score;
                }
            }

            if crawler {
                // Treat blocked crawlers as immovable.
                if board.slidable_adjacent(&mut buf, hex, hex).next().is_none() {
                    immovable.set(hex);
                }
            }

            if node.is_stacked() {
                bug_score *= self.stacked_bug_factor;
            }

            let friendly_queen = board.queens[node.color() as usize];

            // TODO: transpose this loop, i.e. categorize queen liberties after the bug loop.
            // Count libs for more if they are not crawlable (e.g. behind a gate)
            if adjacent(friendly_queen).contains(&hex) {
                // Filling friendly queen's liberty.
                if immovable.get(hex) && !node.is_stacked() {
                    queen_score[node.color() as usize] -= self.queen_liberty_penalty;
                } else {
                    // Lower penalty for being able to leave.
                    queen_score[node.color() as usize] -=
                        self.queen_liberty_penalty / self.queen_movable_penalty_factor;
                    //TODO: /2 in 0.5
                }
                if pillbug_powers
                    && board.node(friendly_queen).clipped_height() == 1
                    && !pinned.get(friendly_queen)
                {
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
                    if best_escape > 2 {
                        pillbug_defense[node.color() as usize] = true;
                    }
                }
            }

            let enemy_queen = board.queens[node.color().other() as usize];

            if adjacent(enemy_queen).contains(&hex) {
                // Discourage liberty filling by valuable bugs, by setting their score to zero when filling a liberty.
                bug_score = 0.0;
                // A little extra boost for filling opponent's queen, as we will never choose to move.
                queen_score[node.color().other() as usize] -=
                    self.queen_liberty_penalty * self.opponent_queen_liberty_penalty_factor;
                if pillbug_powers
                    && board.node(enemy_queen).clipped_height() == 1
                    && !pinned.get(enemy_queen)
                {
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
                    if best_unescape < 3 {
                        queen_score[node.color().other() as usize] -= self.trap_queen_penalty;
                    }
                }
            }

            if !node.is_stacked() && immovable.get(hex) {
                // Pinned bugs are worthless.
                continue;
            }

            if node.bug() == Bug::Ant || mosquito_ant {
                can_pin_beetle[node.color() as usize] = true;
                if node.color() == board.to_move() {
                    //Refactor
                    ant += 1;
                } else {
                    ant_opponent += 1;
                }
            }

            if node.color() != board.to_move() {
                bug_score = -bug_score;
            }
            score += bug_score;
        }

        let ant_difference = ant - ant_opponent;
        score += self.ant_game_factor * ant_difference as f32;

        let mut pillbug_defense_score = 0.0;
        if pillbug_defense[board.to_move() as usize] {
            pillbug_defense_score += self.pillbug_defense_bonus
        }
        if pillbug_defense[board.to_move().other() as usize] {
            pillbug_defense_score -= self.pillbug_defense_bonus;
        }

        // Check for backup defensive pillbug placeability option, discounted value
        pillbug_defense = [false; 2];
        for &color in &[Color::Black, Color::White] {
            //It's possible we place a pillbug but it doesn't have free space to defend the queen.
            if board.node(board.queens[color as usize]).clipped_height() == 1
                && board.remaining[color as usize][Bug::Pillbug as usize] > 0
                && adjacent(board.queens[color as usize])
                    .iter()
                    .any(|&lib| placeable(board, lib, color))
            {
                pillbug_defense[color as usize] = true;
            }
        }

        if pillbug_defense[board.to_move() as usize] {
            pillbug_defense_score += self.placeable_pillbug_defense_bonus
        }
        if pillbug_defense[board.to_move().other() as usize] {
            pillbug_defense_score -= self.placeable_pillbug_defense_bonus;
        }

        // Check for gates.
        //try to do a more spefic check: check if there ara grasshopper that can jump in or there are free grasshopper
        gates_score[board.to_move() as usize] += check_gates(board, board.to_move()) as f32
            * (4 - count_free_grasshoppers(board, board.to_move().other(), &immovable)) as f32;
        gates_score[board.to_move().other() as usize] -= check_gates(board, board.to_move().other())
            as f32
            * (4 - count_free_grasshoppers(board, board.to_move(), &immovable)) as f32;
        let gates_score = (gates_score[board.to_move() as usize]
            - gates_score[board.to_move().other() as usize])
            * self.gates_factor;

        // Check for spawn points next to opponent queen
        // before check if there is an available beetle, otherwise the spawn points will be 0
        //check also if there are bugs that can easily pin our beetle in the future
        let mut queen_spawn_score = 0.0;
        if board.remaining[board.to_move() as usize][Bug::Beetle as usize] > 0 {
            queen_spawn_score = count_queen_spawn_points(board, board.to_move()) as f32;
            if can_pin_beetle[board.to_move().other() as usize] {
                queen_spawn_score /= self.pinnable_beetle_factor;
            }
        }
        let mut queen_spawn_score_opponent = 0.0;
        if board.remaining[board.to_move().other() as usize][Bug::Beetle as usize] > 0 {
            queen_spawn_score_opponent =
                count_queen_spawn_points(board, board.to_move().other()) as f32;
            if can_pin_beetle[board.to_move() as usize] {
                queen_spawn_score_opponent /= self.pinnable_beetle_factor;
            }
        }

        let queen_spawn_score =
            self.queen_spawn_factor * (queen_spawn_score - queen_spawn_score_opponent);

        let queen_score =
            queen_score[board.to_move() as usize] - queen_score[board.to_move().other() as usize];
        (queen_score
            + pillbug_defense_score
            + score
            + gates_score
            + unplayed_bug_score
            + queen_spawn_score) as Evaluation
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
            if !adjacent(board.queens[board.to_move().other() as usize]).contains(&hex) {
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
            let m = strategy.choose_move(&board);
            assert_eq!(Some(Turn::Move(loc_to_hex((-1, 1)), loc_to_hex((2, 1)))), m);

            let mut strategy = Negamax::new(BasicEvaluator::default(), depth);
            let m = strategy.choose_move(&board);
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
            let m = strategy.choose_move(&board);
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

fn is_there_a_filling_grasshopper(board: &Board, color: Color, hex: Hex) -> bool {
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
                if dist > 1 && jump == hex {
                    return true; // Found a filling grasshopper
                }
            }
        }
    }
    false // No filling grasshopper found
}

fn is_there_a_filling_ladybug(board: &Board, color: Color, hex: Hex) -> bool {
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
    false // No filling ladybug found
}

fn count_free_grasshoppers(board: &Board, color: Color, immovable: &HexSet) -> u8 {
    let mut count = 0;

    // Count grasshoppers already on the board and free to move
    for &hex in board.occupied_hexes[color as usize].iter() {
        let node = board.node(hex);
        if node.bug() == Bug::Grasshopper {
            // Check if the grasshopper is immovable, otherwise count it
            if !immovable.get(hex) {
                count += 1 // Skip immovable grasshoppers
            }
        }
    }

    // Add the number of unplayed grasshoppers
    count += board.remaining[color as usize][Bug::Grasshopper as usize];

    count // Return the total count of free grasshoppers
}

fn check_gates(board: &Board, color: Color) -> u8 {
    //Hex
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
            if is_there_a_filling_grasshopper(board, color.other(), adj)
                || is_there_a_filling_ladybug(board, color.other(), adj)
            {
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
    board.slidable_adjacent(&mut neighbors, origin, hex).collect::<Vec<_>>().is_empty()
    // Otherwise, it's not a gate.
}

////////////////////////////////////////////////////////////////////////////////////////////////

fn spawn_points_flag(board: &Board, color: Color) -> bool {
    for &hex in board.occupied_hexes[color as usize].iter() {
        for adj in adjacent(hex) {
            // Check if the adjacent hex is unoccupied and placeable for the given color
            if !board.occupied(adj) && placeable(board, adj, color) {
                return true; // Found a spawn point
            }
        }
    }
    false // No spawn points found
}

//function to count the spawn points near a peice next to the adversary queen
fn count_queen_spawn_points(board: &Board, color: Color) -> u8 {
    //opponent queen
    let opp_queen = board.queens[1 - (color as usize)]; //not opponent queen
    let mut queen_spawn = 0;
    //check if there are our pieces next to the opponent queen
    for adj in adjacent(opp_queen) {
        //check if the hex is occupied by our pieces
        if board.occupied(adj) && board.node(adj).color() == color {
            //now check if an adjacent hex to this adj is unoccupied and placeable for us
            for space in adjacent(adj) {
                //check if the hex is unoccupied and placeable for us
                if !board.occupied(space) && placeable(board, space, color) {
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
        assert_eq!(check_gates(&board, Color::Black), 1);
        assert!(!is_there_a_filling_grasshopper(&board, Color::White, loc_to_hex((1, 1))));

        let mut board = Board::default();
        board.apply(Turn::Place(loc_to_hex((0, 0)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((1, 0)), Bug::Spider));
        board.apply(Turn::Place(loc_to_hex((-1, 1)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((2, 3)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((1, 2)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((0, 1)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((2, 2)), Bug::Beetle));
        board.apply(Turn::Pass);
        assert!(is_there_a_filling_grasshopper(&board, Color::White, loc_to_hex((1, 1))));
        assert_eq!(check_gates(&board, Color::Black), 0);

        let mut board = Board::default();
        board.apply(Turn::Place(loc_to_hex((0, 0)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((1, 0)), Bug::Spider));
        board.apply(Turn::Place(loc_to_hex((-1, 1)), Bug::Ladybug));
        board.apply(Turn::Place(loc_to_hex((2, 3)), Bug::Ant));
        board.apply(Turn::Place(loc_to_hex((1, 2)), Bug::Grasshopper));
        board.apply(Turn::Place(loc_to_hex((0, 1)), Bug::Queen));
        board.apply(Turn::Place(loc_to_hex((2, 2)), Bug::Beetle));
        board.apply(Turn::Pass);
        assert!(is_there_a_filling_ladybug(&board, Color::White, loc_to_hex((1, 1))));
        assert_eq!(check_gates(&board, Color::Black), 0);
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
        assert!(spawn_points_flag(&board, Color::Black));
    }
}
/*
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
    }
}
*/
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

//TODO: - extra strong queen liberty factor
//      - initial turns
//      - aggression fixed at 5
