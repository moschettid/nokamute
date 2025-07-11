// Board representation: wrapping grid of tile locations.
// Rows wrap around, and each row wraps to the next row.
// It's like a spiral around a torus.
//
// A 4x4 example:
//
//         11  12  13  14  15
//           \ / \ / \ / \ /
//       15 - 0 - 1 - 2 - 3 - 4
//         \ / \ / \ / \ / \
//      3 - 4 - 5 - 6 - 7 - 8
//       \ / \ / \ / \ / \
//    7 - 8 - 9 -10 -11 -12
//     \ / \ / \ / \ / \
// 11 -12 -13 -14 -15 - 0
//     / \ / \ / \ / \
//    0   1   2   3   4

// Index of a board location.
#[cfg(not(feature = "larger-grid"))]
pub(crate) type Hex = u8;
#[cfg(feature = "larger-grid")]
pub(crate) type Hex = u16;

#[cfg(not(feature = "larger-grid"))]
pub const ROW_SIZE: Hex = 16;
#[cfg(feature = "larger-grid")]
pub const ROW_SIZE: Hex = 32;

pub(crate) const GRID_SIZE: usize = ROW_SIZE as usize * ROW_SIZE as usize;
pub(crate) const GRID_MASK: Hex = (GRID_SIZE as Hex).wrapping_sub(1);
// In the middle of the columns and the rows.
// To slightly increase cache locality in the early game,
// and to make formatting slightly simpler.
pub(crate) const START_HEX: Hex = ROW_SIZE / 2 * (ROW_SIZE + 1);

#[repr(u16)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Direction {
    NW = (GRID_MASK & (ROW_SIZE + 1).wrapping_neg()) as u16,
    NE = (GRID_MASK & ROW_SIZE.wrapping_neg()) as u16,
    W = GRID_MASK as u16 & 1u16.wrapping_neg(),
    E = 1,
    SW = ROW_SIZE as u16,
    SE = ROW_SIZE as u16 + 1,
}

// In clockwise order
impl Direction {
    pub fn all() -> &'static [Direction; 6] {
        // Omits None.
        &[Direction::NW, Direction::NE, Direction::E, Direction::SE, Direction::SW, Direction::W]
    }

    pub fn apply(self, hex: Hex) -> Hex {
        GRID_MASK & hex.wrapping_add(self as Hex)
    }
}

pub(crate) fn adjacent(hex: Hex) -> [Hex; 6] {
    // In clockwise order
    [
        Direction::NW.apply(hex),
        Direction::NE.apply(hex),
        Direction::E.apply(hex),
        Direction::SE.apply(hex),
        Direction::SW.apply(hex),
        Direction::W.apply(hex),
    ]
}

#[test]
fn test_direction() {
    // Reversibility from any position.
    for hex in 0..=GRID_MASK {
        assert_eq!(hex, Direction::NE.apply(Direction::SW.apply(hex)));
        assert_eq!(hex, Direction::E.apply(Direction::W.apply(hex)));
        assert_eq!(hex, Direction::SE.apply(Direction::NW.apply(hex)));
    }

    // Exercise the known wrapping properties.
    // Two axes traverse the entire space before wrapping.
    let mut hex = START_HEX;
    for _ in 1..GRID_SIZE {
        hex = Direction::E.apply(hex);
        assert_ne!(hex, START_HEX);
    }
    hex = Direction::E.apply(hex);
    assert_eq!(hex, START_HEX);

    hex = START_HEX;
    for _ in 1..GRID_SIZE {
        hex = Direction::NW.apply(hex);
        assert_ne!(hex, START_HEX);
    }
    hex = Direction::NW.apply(hex);
    assert_eq!(hex, START_HEX);

    // One axis only traverses one row.
    hex = START_HEX;
    for _ in 1..ROW_SIZE {
        hex = Direction::NE.apply(hex);
        assert_ne!(hex, START_HEX);
    }
    hex = Direction::NE.apply(hex);
    assert_eq!(hex, START_HEX);
}

// Efficient set utility.
const HEXSET_NUM_WORDS: usize = GRID_SIZE / 32;
const HEXSET_SHIFT: u32 = GRID_SIZE.trailing_zeros() - 5;
const HEXSET_MASK: usize = HEXSET_NUM_WORDS - 1;

pub(crate) struct HexSet {
    table: [u32; HEXSET_NUM_WORDS],
}

impl HexSet {
    pub(crate) fn new() -> HexSet {
        HexSet { table: [0; HEXSET_NUM_WORDS] }
    }

    pub(crate) fn set(&mut self, hex: Hex) {
        self.table[hex as usize & HEXSET_MASK] |= 1 << (hex as u32 >> HEXSET_SHIFT);
    }

    pub(crate) fn get(&self, hex: Hex) -> bool {
        (self.table[hex as usize & HEXSET_MASK] >> (hex as u32 >> HEXSET_SHIFT)) & 1 != 0
    }
}

//TO DO: REFACTOR
pub(crate) fn find_aligned_hex(hex1: Hex, hex2: Hex) -> Option<Hex> { //consecuitevely aligned hexes
    // Controlla tutte le 6 direzioni da hex1
    for &direction in Direction::all() { 
        let adjacent_hex = direction.apply(hex1);
        
        // Se hex2 è adiacente a hex1 in questa direzione
        if adjacent_hex == hex2 {
            // Il terzo hex allineato sarà nella stessa direzione da hex2
            let aligned_hex = direction.apply(hex2);
            return Some(aligned_hex);
        }
    }
    None
}

pub (crate) fn is_aligned(hex1: Hex, hex2: Hex, hex3: Hex) -> bool { // if h3 is aligned with h1 and h2, consecutively
    // Controlla se hex3 è allineato con hex1 e hex2
    if let Some(expected_hex3) = find_aligned_hex(hex1, hex2) { //hex1, hex2, hex3
        return expected_hex3 == hex3;
    }
    
    // Controlla anche nell'ordine inverso (hex2, hex1, hex3)
    if let Some(expected_hex3) = find_aligned_hex(hex2, hex1) { //hex2, hex1, hex3
        return expected_hex3 == hex3;
    }
    
    false
}

pub(crate) fn forms_triangle(hex1: Hex, hex2: Hex, hex3: Hex) -> bool { 
    let adj1 = adjacent(hex1);
    let adj2 = adjacent(hex2);
    let adj3 = adjacent(hex3);
    
    // Ogni hex deve essere nell'array di adiacenze degli altri due
    adj1.contains(&hex2) && adj1.contains(&hex3) && 
    adj2.contains(&hex1) && adj2.contains(&hex3) && 
    adj3.contains(&hex1) && adj3.contains(&hex2)
}