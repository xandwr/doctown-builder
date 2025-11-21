// Test file for enhanced symbol resolution features

// 1. TYPE INFERENCE TEST
pub fn create_user() -> User {
    User { name: String::from("Alice") }
}

pub fn type_inference_example() {
    // Type inferred from function call
    let user = create_user();  // Should infer User type

    // Explicit type annotation
    let count: usize = 42;
}

// 2. TRAIT RESOLUTION TEST
pub trait Drawable {
    fn draw(&self);
    fn color(&self) -> String;
}

pub struct Circle {
    radius: f64,
}

// Trait implementation - should create TraitImplementation edge
impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle");
    }

    fn color(&self) -> String {
        String::from("red")
    }
}

// 3. METHOD DISPATCH TEST
pub struct User {
    name: String,
}

impl User {
    pub fn new(name: String) -> Self {
        User { name }
    }

    pub fn greet(&self) {
        println!("Hello, {}", self.name);
    }
}

pub fn method_dispatch_example() {
    let user = User::new(String::from("Bob"));
    user.greet();  // Should resolve to User::greet method

    let circle = Circle { radius: 5.0 };
    circle.draw();  // Should resolve to Circle::draw (trait method)
}

// 4. MACRO EXPANSION TEST
macro_rules! debug_print {
    ($x:expr) => {
        println!("Debug: {:?}", $x);
    };
}

pub fn macro_example() {
    debug_print!("test");  // Should create MacroExpansion edge
    println!("Standard macro");  // Standard library macro
}

// Additional complex example combining multiple features
pub trait Container<T> {
    fn add(&mut self, item: T);
    fn get(&self, index: usize) -> Option<&T>;
}

pub struct MyVec<T> {
    items: Vec<T>,
}

impl<T> Container<T> for MyVec<T> {
    fn add(&mut self, item: T) {
        self.items.push(item);
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }
}

pub fn complex_example() {
    // Type inference with generics
    let mut container = MyVec { items: vec![1, 2, 3] };
    container.add(4);  // Method dispatch on generic type

    // Trait method call
    let item = container.get(0);
    debug_print!(item);
}
