extern crate minimax;

use crate::adjacent;
#[cfg(not(target_arch = "wasm32"))]
use crate::cli::CliPlayer;
use crate::hex_grid::forms_triangle;
use crate::hex_grid::is_aligned;
#[cfg(not(target_arch = "wasm32"))]
use crate::mcts::BiasedRollouts;
#[cfg(not(target_arch = "wasm32"))]
use crate::uhp_client::UhpPlayer;
use crate::{nokamute_version, BasicEvaluator, Board, Bug, Rules, Turn};
use minimax::*;
use std::time::Duration;

//TO DO: check if thread and depth are hardcoded or not, look for "HARDCODED" key word

//TO DO:
// - add a probability distribution to the openings
// - check if is right to align to 1st and 3rd and not other
// - check queen position onn opening
// - try string moves approach

// A player that can play one color's moves.
pub(crate) trait Player {
    fn name(&self) -> String;
    fn new_game(&mut self, game_type: &str);
    fn play_move(&mut self, m: Turn);
    fn undo_move(&mut self, m: Turn);
    fn generate_move(&mut self) -> Turn;
    fn principal_variation(&self) -> Vec<Turn> {
        Vec::new()
    }
    fn set_max_depth(&mut self, _depth: u8) {}
    fn set_timeout(&mut self, _time: Duration) {}
}

#[cfg(not(target_arch = "wasm32"))]
fn face_off(
    game_type: &str, mut player1: Box<dyn Player>, mut player2: Box<dyn Player>,
) -> Option<String> {
    let mut b = Board::from_game_type(game_type).unwrap();
    player1.new_game(game_type);
    player2.new_game(game_type);
    let mut players = [player1, player2];
    let mut p = 0;
    loop {
        b.println();
        println!("{} ({:?}) to move", players[p].name(), b.to_move());
        let m = players[p].generate_move();
        let mut moves = Vec::new();
        Rules::generate_moves(&b, &mut moves);
        if !moves.contains(&m) {
            println!("{} played an illegal move: {}", players[p].name(), b.to_move_string(m));
            println!("Game log: {}", b.game_log());
            return Some(players[1 - p].name());
        }
        b.apply(m);
        if let Some(winner) = Rules::get_winner(&b) {
            b.println();
            println!("Game log: {}", b.game_log());
            return match winner {
                minimax::Winner::Draw => None,
                minimax::Winner::PlayerJustMoved => Some(players[p].name()),
                minimax::Winner::PlayerToMove => Some(players[1 - p].name()),
            };
        }
        players[p].play_move(m);
        p = 1 - p;
        players[p].play_move(m);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn get_player(name: &str, config: &PlayerConfig) -> Box<dyn Player> {
    match name {
        "nokamute" => config.new_player(),
        "ai" => config.new_player(),
        "human" => Box::new(CliPlayer::new()),
        // Try to launch this as a UHP server
        _ => Box::new(UhpPlayer::new(name).unwrap()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn play_game(
    //TODO: should support double config
    config: PlayerConfig,
    game_type: &str,
    name1: &str,
    name2: &str,
    depth: Option<u8>,
    timeout: Option<String>,
) {
    let mut player1 = get_player(name1, &config);
    let mut player2 = get_player(name2, &config);
    if let Some(depth) = depth {
        player1.set_max_depth(depth);
        player2.set_max_depth(depth);
    } else if let Some(input) = timeout {
        let timeout = if input.ends_with('s') {
            input[..input.len() - 1].parse::<u64>().map(Duration::from_secs)
        } else if input.ends_with('m') {
            input[..input.len() - 1].parse::<u64>().map(|m| Duration::from_secs(m * 60))
        } else {
            exit("Could not parse --timeout (add units)".to_string());
        }
        .unwrap_or_else(|_| exit("Could not parse --timeout (add units)".to_string()));
        player1.set_timeout(timeout);
        player2.set_timeout(timeout);
    }
    match face_off(game_type, player1, player2) {
        None => println!("Game over: draw."),
        Some(name) => println!("Game over: {} won.", name),
    }
}

struct NokamutePlayer {
    board: Board,
    strategy: Box<dyn Strategy<Rules> + Send>,
    random_opening: bool,
    name: String,
    actual_opening: Option<(String, Vec<Bug>)>,
}

impl NokamutePlayer {
    fn new(strategy: Box<dyn Strategy<Rules> + Send>, random_opening: bool) -> Self {
        Self::new_with_name(&format!("nokamute {}", nokamute_version()), strategy, random_opening)
    }

    fn new_with_name(
        name: &str, mut strategy: Box<dyn Strategy<Rules> + Send>, random_opening: bool,
    ) -> Self {
        strategy.set_timeout(Duration::from_secs(5));
        NokamutePlayer {
            board: Board::default(),
            strategy,
            random_opening,
            name: name.to_owned(),
            actual_opening: None,
        }
    }
}

impl Player for NokamutePlayer {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn new_game(&mut self, game_string: &str) {
        self.board = Board::from_game_string(game_string).unwrap();
    }

    fn play_move(&mut self, m: Turn) {
        self.board.apply(m);
    }

    fn undo_move(&mut self, m: Turn) {
        self.board.undo(m);
    }

    fn generate_move(&mut self) -> Turn {
        //Hardcoded Antispawn opening
        //I want to hardcode the first 3 moves as Ladybug, Queen, Pillbug
        // if self.board.turn_num < 2 {
        //     //if it's white
        //     if self.board.to_move() == Color::White {
        //         // If it's the first move, place a Ladybug
        //         return Turn::Place(loc_to_hex((0, 0)), Bug::Ladybug);
        //     }else {
        //         // If it's the first move, place a Beetle
        //         return Turn::Place(loc_to_hex((0, 1)), Bug::Ladybug);
        //     }

        // } else if self.board.turn_num < 4 {
        //     if self.board.to_move() == Color::White {
        //         // If it's the first move, place a Ladybug
        //         return Turn::Place(loc_to_hex((1, 0)), Bug::Queen);
        //     }else {
        //         // If it's the first move, place a Beetle
        //         return Turn::Place(loc_to_hex((1, 2)), Bug::Queen);
        //     }
        // } else if self.board.turn_num < 6 {
        //     if self.board.to_move() == Color::White {
        //         return Turn::Place(loc_to_hex((2, 0)), Bug::Pillbug);
        //     }else {
        //         // If it's the first move, place a Beetle
        //         return Turn::Place(loc_to_hex((2, 3)), Bug::Pillbug);
        //     }
        // }
        //////////////////////////////////////////////////////////////////////////////////
        //version that implement a single opening functioning
        // if true {//self.random_opening
        //     // Ignore minimax and just throw out a random jumpy bug for the first move.
        //     if self.board.turn_num < 2 {
        //         loop {
        //             let turn =
        //                 minimax::Random::<Rules>::default().choose_move(&self.board).unwrap();
        //             if let Turn::Place(hex, bug) = turn {
        //                 if matches!(
        //                     bug,
        //                     Bug::Ladybug // | Bug::Beetle | Bug::Grasshopper
        //                 ) {
        //                     // let mut moves = Vec::new();
        //                     // Rules::generate_moves(&self.board, &mut moves);
        //                     //I want to print all the moves
        //                     //println!("Possible moves: {:?}", moves);
        //                     println!("{:?}", turn);
        //                     return turn;
        //                 }
        //             }
        //         }
        //     } else if self.board.turn_num < 4 {
        //         loop {
        //             let turn =
        //                 minimax::Random::<Rules>::default().choose_move(&self.board).unwrap();
        //             if let Turn::Place(_, bug) = turn {
        //                 if bug == Bug::Queen {
        //                     return turn;
        //                 }
        //             }
        //         }
        //     }
        //     if self.board.turn_num < 6 {
        //         loop {
        //             let turn =
        //                 minimax::Random::<Rules>::default().choose_move(&self.board).unwrap();
        //             if let Turn::Place(_, bug) = turn {
        //                 //I want to check is a pillbug and if it is aligned with the queen and ladybug
        //                 if bug == Bug::Pillbug {
        //                     let mut moves = Vec::new();
        //                     Rules::generate_moves(&self.board, &mut moves);
        //                     let current_color = self.board.to_move();
        //                     let queen_hex = self.board.find_bug(current_color, Bug::Queen, 1);
        //                     let ladybug_hex = self.board.find_bug(current_color, Bug::Ladybug, 1);
        //                     // println!("length: {}", moves.len());
        //                     // println!("moves: {:?}", moves);
        //                     //iterate on all moves and check if the pillbug is aligned with the queen and ladybug
        //                     for m in moves {
        //                         if let Turn::Place(hex, bug) = m {
        //                             if bug == Bug::Pillbug {
        //                                 let pillbug_hex = hex;
        //                                 // Debug: stampa le posizioni trovate
        //                                 println!("Ladybug hex: {:?}", ladybug_hex);
        //                                 println!("Queen hex: {:?}", queen_hex);
        //                                 println!("Pillbug hex: {:?}", pillbug_hex);

        //                                 // Controlla se abbiamo sia Queen che Ladybug
        //                                 if let (Some(q_hex), Some(l_hex)) = (queen_hex, ladybug_hex) {
        //                                     // Controlla se sono allineati
        //                                     if is_aligned(pillbug_hex, q_hex, l_hex) {
        //                                         println!("Pillbug allineato trovato!");
        //                                         println!("Mossa allineata: {:?}", m);
        //                                         return m; // Ritorna solo la mossa allineata
        //                                     }
        //                                 }
        //                             }
        //                         }
        //                     }
        //                     return turn;
        //                 }
        //             }
        //         }
        //     }
        // }
        //////////////////////////////////////////////////////////////////////////////////
        if self.board.turn_num < 2 {
            let openings = Openings::new();
            let distribution = [0.4, 0.1, 0.1, 0.1, 0.3, 0.0, 0.0, 0.0];
            let chosen_opening = openings.get_random_opening(Some(&distribution));

            if let Some((opening_name, bug_vector)) = chosen_opening {
                self.actual_opening = Some((opening_name.clone(), bug_vector.clone()));
                return openings.get_opening(opening_name, &self.board, bug_vector).unwrap();
            }
        }
        if self.board.turn_num < 6 {
            if let Some((opening_name, bug_vector)) = &self.actual_opening {
                let openings = Openings::new(); // Devi creare una nuova istanza qui
                return openings.get_opening(opening_name, &self.board, bug_vector).unwrap();
            }
        }
        if self.board.turn_num < 8 {
            if let Some((opening_name, bug_vector)) = &self.actual_opening {
                if bug_vector.len() > 3 {
                    let openings = Openings::new(); // Devi creare una nuova istanza qui
                    let fourth_move =
                        openings.get_opening(opening_name, &self.board, bug_vector).unwrap();
                    //check if fourth move is None
                    if let Some(Turn::Place(_h, _b)) = Some(fourth_move) {
                        return Turn::Place(_h, _b);
                    }
                }
            }
        }
        self.strategy.choose_move(&self.board).unwrap()
    }

    fn principal_variation(&self) -> Vec<Turn> {
        self.strategy.principal_variation()
    }

    fn set_max_depth(&mut self, depth: u8) {
        self.strategy.set_max_depth(depth);
    }

    fn set_timeout(&mut self, time: Duration) {
        self.strategy.set_timeout(time);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn exit(msg: String) -> ! {
    eprintln!("{}", msg);
    std::process::exit(1)
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub(crate) enum PlayerStrategy {
    Iterative(ParallelOptions),
    Random,
    Mcts(MCTSOptions),
}

#[derive(Clone)]
pub struct PlayerConfig {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) num_threads: Option<usize>,
    pub(crate) opts: IterativeOptions,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) strategy: PlayerStrategy,
    pub(crate) eval: BasicEvaluator,
    pub(crate) random_opening: bool,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn configure_player() -> Result<(PlayerConfig, Vec<String>), pico_args::Error> {
    let mut args = pico_args::Arguments::from_env();

    let mut config = PlayerConfig::new();

    // Configure common minimax options.
    if args.contains(["-v", "--verbose"]) {
        config.opts = config.opts.verbose();
    }
    let table_size: Option<usize> = args.opt_value_from_str("--table_mb")?;
    if let Some(table_size) = table_size {
        config.opts.table_byte_size = table_size.checked_shl(20).unwrap();
    }
    let window_arg: Option<u32> = args.opt_value_from_str("--aspiration-window")?;
    if let Some(window) = window_arg {
        config.opts = config.opts.with_aspiration_window(window as minimax::Evaluation);
    }
    if args.contains("--double-step") {
        config.opts = config.opts.with_double_step_increment();
    }
    if args.contains("--null-move-pruning") {
        config.opts = config.opts.with_null_move_depth(3);
    }
    if args.contains("--quiet-search") {
        config.opts = config.opts.with_quiescence_search_depth(2);
    }

    let eval_string: Option<String> = args.opt_value_from_str("--set-eval")?;
    if let Some(eval_string) = eval_string {
        let mut evaluator = BasicEvaluator::new();

        for param_pair in eval_string.split(',') {
            let parts: Vec<&str> = param_pair.trim().split(':').collect();
            if parts.len() != 2 {
                exit(format!("Invalid parameter format in --set-eval: {}", param_pair));
            }

            let param_name = parts[0].trim();
            let param_value = parts[1].trim().parse::<f32>().unwrap_or_else(|_| {
                exit(format!("Could not parse value for parameter {}: {}", param_name, parts[1]))
            });

            // Apply the appropriate setter based on parameter name
            match param_name {
                "queen_liberty_penalty" => {
                    evaluator.queen_liberty_penalty(param_value);
                }
                "gates_factor" => {
                    evaluator.gates_factor(param_value);
                }
                "queen_spawn_factor" => {
                    evaluator.queen_spawn_factor(param_value);
                }
                "unplayed_bug_factor" => {
                    evaluator.unplayed_bug_factor(param_value);
                }
                "pillbug_defense_bonus" => {
                    evaluator.pillbug_defense_bonus(param_value);
                }
                "ant_game_factor" => {
                    evaluator.ant_game_factor(param_value);
                }
                "queen_score" => {
                    evaluator.queen_score(param_value);
                }
                "ant_score" => {
                    evaluator.ant_score(param_value);
                }
                "beetle_score" => {
                    evaluator.beetle_score(param_value);
                }
                "grasshopper_score" => {
                    evaluator.grasshopper_score(param_value);
                }
                "spider_score" => {
                    evaluator.spider_score(param_value);
                }
                "mosquito_score" => {
                    evaluator.mosquito_score(param_value);
                }
                "ladybug_score" => {
                    evaluator.ladybug_score(param_value);
                }
                "pillbug_score" => {
                    evaluator.pillbug_score(param_value);
                }
                "mosquito_incremental_score" => {
                    evaluator.mosquito_incremental_score(param_value);
                }
                "stacked_bug_factor" => {
                    evaluator.stacked_bug_factor(param_value);
                }
                "queen_movable_penalty_factor" => {
                    evaluator.queen_movable_penalty_factor(param_value);
                }
                "opponent_queen_liberty_penalty_factor" => {
                    evaluator.opponent_queen_liberty_penalty_factor(param_value);
                }
                "trap_queen_penalty" => {
                    evaluator.trap_queen_penalty(param_value);
                }
                "placeable_pillbug_defense_bonus" => {
                    evaluator.placeable_pillbug_defense_bonus(param_value);
                }
                "pinnable_beetle_factor" => {
                    evaluator.pinnable_beetle_factor(param_value);
                }
                //new params
                "mosquito_ant_factor" => {
                    evaluator.mosquito_ant_factor(param_value);
                }
                "mobility_factor" => {
                    evaluator.mobility_factor(param_value);
                }
                "compactness_factor" => {
                    evaluator.compactness_factor(param_value);
                }
                "pocket_factor" => {
                    evaluator.pocket_factor(param_value);
                }
                "beetle_attack_factor" => {
                    evaluator.beetle_attack_factor(param_value);
                }
                "beetle_on_enemy_queen_factor" => {
                    evaluator.beetle_on_enemy_queen_factor(param_value);
                }
                "beetle_on_enemy_pillbug_factor" => {
                    evaluator.beetle_on_enemy_pillbug_factor(param_value);
                }
                "direct_queen_drop_factor" => {
                    evaluator.direct_queen_drop_factor(param_value);
                }
                _ => exit(format!("Unknown parameter in --set-eval: {}", param_name)),
            };
        }

        config.eval = evaluator;
    }

    // 0 for num_cpu threads; >0 for specific count.
    // Always use 1 thread for num_threads.
    //THREAD HARDCODED
    config.num_threads = Some(1);
    // Previous code for parsing --num-threads:
    // config.num_threads = args.opt_value_from_str("--num-threads")?.map(|thread_arg: String| {
    //     if thread_arg == "max" || thread_arg == "all" {
    //         0
    //     } else if let Ok(num) = thread_arg.parse::<usize>() {
    //         num
    //     } else {
    //         exit(format!("Could not parse num_threads={}. Expected int or 'max'", thread_arg));
    //     }
    // });

    // Configure specific strategy.
    let strategy: Option<String> = args.opt_value_from_str("--strategy")?;
    config.strategy = match strategy.as_deref().unwrap_or("iterative") {
        "random" => PlayerStrategy::Random,
        "mcts" => {
            let mut options = MCTSOptions::default()
                .with_max_rollout_depth(200)
                .with_rollouts_before_expanding(5);
            options.verbose = config.opts.verbose;
            PlayerStrategy::Mcts(options)
        }
        "mtdf" => {
            config.opts = config.opts.with_mtdf();
            config.num_threads = Some(1);
            PlayerStrategy::Iterative(ParallelOptions::new())
        }
        "iterative" => {
            let mut parallel_opts = ParallelOptions::new();
            if args.contains("--background-ponder") {
                parallel_opts = parallel_opts.with_background_pondering();
            }
            PlayerStrategy::Iterative(parallel_opts)
        }
        _ => exit(format!("Unrecognized strategy: {}", strategy.unwrap_or_default())),
    };
    Ok((config, args.finish().into_iter().map(|s| s.into_string().unwrap()).collect::<Vec<_>>()))
}

impl Default for PlayerConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl PlayerConfig {
    pub fn new() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            num_threads: None,
            opts: IterativeOptions::new()
                .with_countermoves()
                .with_countermove_history()
                .with_table_byte_size(100 << 20),
            #[cfg(not(target_arch = "wasm32"))]
            strategy: PlayerStrategy::Iterative(ParallelOptions::new()),
            eval: BasicEvaluator::default(),
            random_opening: false,
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn new_player(&self) -> Box<dyn Player + Send> {
        Box::new(NokamutePlayer::new(
            Box::new(IterativeSearch::new(self.eval, self.opts)),
            self.random_opening,
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn new_player(&self) -> Box<dyn Player + Send> {
        Box::new(match &self.strategy {
            PlayerStrategy::Random => NokamutePlayer::new_with_name(
                "random",
                Box::<Random<Rules>>::default(),
                self.random_opening,
            ),
            PlayerStrategy::Mcts(opts) => {
                let mut opts = opts.clone();
                let num_threads = self.num_threads.unwrap_or(0);
                if num_threads > 0 {
                    opts = opts.with_num_threads(num_threads);
                }
                NokamutePlayer::new(
                    Box::new(MonteCarloTreeSearch::new_with_policy(
                        opts,
                        Box::new(BiasedRollouts {}),
                    )),
                    self.random_opening,
                )
            }
            PlayerStrategy::Iterative(parallel_opts) => {
                let mut parallel_opts = *parallel_opts;
                let num_threads = self.num_threads.unwrap_or(0);
                if num_threads > 0 {
                    parallel_opts = parallel_opts.with_num_threads(num_threads);
                }
                NokamutePlayer::new(
                    if num_threads == 1 {
                        Box::new(IterativeSearch::new(self.eval, self.opts))
                    } else {
                        Box::new(ParallelSearch::new(self.eval, self.opts, parallel_opts))
                    },
                    self.random_opening,
                )
            }
        })
    }
}

//Openings structure, that if it's initiate return a random opening from a list, if it's called with an identifier of the opening and the board instead return the next move of the opening

pub struct Openings {
    openings: Vec<(String, Vec<Bug>)>,
}

impl Default for Openings {
    fn default() -> Self {
        Openings::new()
    }
}

impl Openings {
    pub fn new() -> Self {
        let openings = vec![
            (
                "antispawn".to_string(),
                // Bug vector with Ladybug, Queen, Pillbug
                vec![Bug::Ladybug, Bug::Queen, Bug::Pillbug],
            ),
            (
                "antispawn first variation".to_string(),
                // Grasshopper, Queen, Pillbug
                vec![Bug::Grasshopper, Bug::Queen, Bug::Pillbug],
            ),
            //NOTE: this two variation are okay if the adversary doesn't play the ant
            (
                "antispawn second variation".to_string(),
                // Ladybug, Queen, Ant, Pillbug
                vec![Bug::Ladybug, Bug::Queen, Bug::Ant, Bug::Pillbug],
            ),
            (
                "antispawn third variation".to_string(),
                // Ladybug, Queen, Ant, Pillbug
                vec![Bug::Ladybug, Bug::Queen, Bug::Mosquito, Bug::Pillbug],
            ),
            (
                "diamond".to_string(),
                // Ladybug, Queen, Ant, Mosquito
                vec![Bug::Ladybug, Bug::Queen, Bug::Ant, Bug::Mosquito],
            ),
            (
                "quick ant attack".to_string(),
                // Pillbug, Queen, Ant
                vec![Bug::Pillbug, Bug::Queen, Bug::Ant],
            ),
            (
                "quick ant attack first variation".to_string(),
                // Ladybug, Queen, Ant
                vec![Bug::Ladybug, Bug::Queen, Bug::Ant],
            ),
            (
                "quick ant attack second variation".to_string(),
                // Grasshopper, Queen, Ant
                vec![Bug::Grasshopper, Bug::Queen, Bug::Ant],
            ),
        ];
        Openings { openings }
    }
    //distribution as parameter, 8 element vector
    pub fn get_random_opening(&self, distribution: Option<&[f64]>) -> Option<(&String, &Vec<Bug>)> {
        // if self.openings.is_empty() {
        //     return None;
        // }
        // use rand::Rng;
        // let mut rng = rand::thread_rng();
        // let index = rng.gen_range(0..self.openings.len());
        // Some((&self.openings[index].0, &self.openings[index].1))

        use rand::seq::index;
        use rand::thread_rng;

        let default_distribution = [0.4, 0.1, 0.1, 0.1, 0.3, 0.0, 0.0, 0.0];
        let weights = distribution.unwrap_or(&default_distribution);

        let mut rng = thread_rng();

        //test of the distribution: generate 50 values and print them
        // let mut counts = vec![0.0; self.openings.len()];
        // for _ in 0..50 {
        //     let index = index::sample_weighted(&mut rng, self.openings.len(), |i| weights.get(i).copied().unwrap_or(0.0), 1);
        //     if let Ok(indices) = index {
        //         if let Some(selected_index) = indices.iter().next() {
        //             //println!("Selected index: {}", selected_index);
        //             counts[selected_index] += 1.0;
        //         }
        //     }
        // }
        // // Print the counts for each index
        // for (i, count) in counts.iter().enumerate() {
        //     println!("Opening {}: {}", i, count);
        // }

        // Usa sample_weighted per selezionare un singolo indice
        match index::sample_weighted(
            &mut rng,
            self.openings.len(),
            |i| weights.get(i).copied().unwrap_or(0.0),
            1,
        ) {
            Ok(indices) => {
                // Prendi il primo (e unico) indice selezionato
                if let Some(selected_index) = indices.iter().next() {
                    Some((&self.openings[selected_index].0, &self.openings[selected_index].1))
                } else {
                    None
                }
            }
            Err(_) => {
                // Fallback a selezione casuale uniforme in caso di errore
                use rand::Rng;
                let index = rng.gen_range(0..self.openings.len());
                Some((&self.openings[index].0, &self.openings[index].1))
            }
        }
    }
    pub fn get_opening(&self, identifier: &str, board: &Board, bug_vector: &[Bug]) -> Option<Turn> {
        // for (name, moves) in &self.openings {
        //     if name == identifier && turn_num < moves.len() {
        //         let turn = moves[turn_num].clone();
        //         // Check if the move is valid on the current board
        //         if Rules::is_valid_move(board, &turn) {
        //             return Some(turn);
        //         }
        //     }
        // }
        // None
        if board.turn_num < 2 {
            loop {
                let turn = minimax::Random::<Rules>::default().choose_move(board).unwrap();
                if let Turn::Place(_, bug) = turn {
                    if bug == bug_vector[0] {
                        // Rimuovi matches! se controlli solo un bug
                        return Some(turn); // Ritorna Some(turn)
                    }
                }
            }
        } else if board.turn_num < 4 {
            // let turn =
            //     minimax::Random::<Rules>::default().choose_move(&board).unwrap();
            // if let Turn::Place(_, bug) = turn {
            //     if bug == Bug::Queen { //hardcoded as in all openings the queen is the second bug
            //         return Some(turn);
            //     }
            // }//TO DO: if diamond I need the queen in a specific position, not aligned with the piece of the adversary
            if identifier == "diamond" {
                let mut moves = Vec::new();
                Rules::generate_moves(board, &mut moves);
                let current_color = board.to_move();
                let first_hex = board.find_bug(current_color, bug_vector[0], 1);
                //find the adversary piece near our first bug
                let mut adversary_hex = None;
                for adj in adjacent(first_hex.unwrap()) {
                    if board.occupied(adj) {
                        adversary_hex = Some(adj);
                    }
                }
                for m in moves {
                    if let Turn::Place(hex, bug) = m {
                        if bug == bug_vector[1] {
                            let third_hex = hex;
                            if let (Some(q_hex), Some(l_hex)) = (adversary_hex, first_hex) {
                                // Check not alignment
                                if !is_aligned(q_hex, l_hex, third_hex) {
                                    return Some(m);
                                }
                            }
                        }
                    }
                }
            } else {
                let mut moves = Vec::new();
                Rules::generate_moves(board, &mut moves);
                for m in moves {
                    if let Turn::Place(_, bug) = m {
                        if bug == bug_vector[1] {
                            return Some(m);
                        }
                    }
                }
            }
        } else if board.turn_num < 6 {
            //here start the true logic of the opening
            // - if identifier is "antispawn", "antispawn fist variation", "quick ant attack"(or one of its varitions) then I want to check for alignement
            // - otherwise I need to do 4 moves
            //   - if identifier is "diamond" or "diamond first variation" then I want that third bug form a V with the other two and the fourth complete the diamond
            //   - otherwise ("antispawn second variation", "antispawn third variation") I want that third bug form a triangle with the other two and the fourth alignes with the first two
            if identifier == "antispawn"
                || identifier == "antispawn first variation"
                || identifier == "quick ant attack"
                || identifier == "quick ant attack first variation"
                || identifier == "quick ant attack second variation"
            {
                let mut moves = Vec::new();
                Rules::generate_moves(board, &mut moves);
                let current_color = board.to_move();
                let first_hex = board.find_bug(current_color, bug_vector[0], 1);
                let second_hex = board.find_bug(current_color, bug_vector[1], 1);
                for m in moves {
                    if let Turn::Place(hex, bug) = m {
                        if bug == bug_vector[2] {
                            let third_hex = hex;
                            if let (Some(q_hex), Some(l_hex)) = (first_hex, second_hex) {
                                // Check alignment
                                if is_aligned(q_hex, l_hex, third_hex) {
                                    return Some(m);
                                }
                            }
                        }
                    }
                }
            }
            if identifier == "diamond"
                || identifier == "antispawn second variation"
                || identifier == "antispawn third variation"
            {
                let mut moves = Vec::new();
                Rules::generate_moves(board, &mut moves);
                let current_color = board.to_move();
                let first_hex = board.find_bug(current_color, bug_vector[0], 1);
                let second_hex = board.find_bug(current_color, bug_vector[1], 1);
                for m in moves {
                    if let Turn::Place(hex, bug) = m {
                        if bug == bug_vector[2] {
                            let third_hex = hex;
                            // Controlla se abbiamo sia Queen che Ladybug
                            if let (Some(q_hex), Some(l_hex)) = (first_hex, second_hex) {
                                // Check if they form a triangle and if it's a valid move
                                if forms_triangle(q_hex, l_hex, third_hex) {
                                    return Some(m);
                                }
                            }
                        }
                    }
                }
            }
        } else if board.turn_num < 8 {
            if identifier == "diamond" {
                let mut moves = Vec::new();
                Rules::generate_moves(board, &mut moves);
                let current_color = board.to_move();
                let first_hex = board.find_bug(current_color, bug_vector[0], 1);
                let third_hex = board.find_bug(current_color, bug_vector[2], 1);
                for m in moves {
                    if let Turn::Place(hex, bug) = m {
                        if bug == bug_vector[3] {
                            let fourth_hex = hex;
                            // Check if form a triangle with the first and the third bug
                            if let (Some(q_hex), Some(t_hex)) = (first_hex, third_hex) {
                                // Check if they form a triangle and if it's a valid move
                                if forms_triangle(fourth_hex, q_hex, t_hex) {
                                    return Some(m);
                                }
                            }
                        }
                    }
                } //TO DO: check if is right to align to 1st and 3rd and not other
            }
            if identifier == "antispawn second variation"
                || identifier == "antispawn third variation"
            {
                let mut moves = Vec::new();
                Rules::generate_moves(board, &mut moves);
                let current_color = board.to_move();
                let first_hex = board.find_bug(current_color, bug_vector[0], 1);
                let second_hex = board.find_bug(current_color, bug_vector[1], 1);
                for m in moves {
                    if let Turn::Place(hex, bug) = m {
                        if bug == bug_vector[3] {
                            let fourth_hex = hex;
                            // Check if the fourth is aligned with the first two bugs
                            if let (Some(q_hex), Some(l_hex)) = (first_hex, second_hex) {
                                if is_aligned(q_hex, l_hex, fourth_hex) {
                                    return Some(m);
                                }
                            }
                        }
                    }
                }
            }
        }
        // If we reach here, it means we didn't find a valid move in the opening
        // We can return a random ant on the board
        // loop {
        //     let turn =
        //         minimax::Random::<Rules>::default().choose_move(board).unwrap();
        //     if let Turn::Place(hex, bug) = turn {
        //         if bug == Bug::Ant {
        //             println!("No valid opening move found, returning a random ant: {:?}", turn);
        //             println!("Current opening: {:?}", identifier);
        //             return Some(turn);
        //         }
        //     }
        // }
        Some(Turn::Pass) // If no valid move is found, return Pass
    }
}
