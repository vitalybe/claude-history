use std::{fs, path::Path};

#[test]
fn semantic_feature_gate_is_removed() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest = fs::read_to_string(manifest_path).expect("read Cargo.toml");
    let parsed: toml::Value = toml::from_str(&manifest).expect("parse Cargo.toml");
    let dependencies = parsed
        .get("dependencies")
        .and_then(toml::Value::as_table)
        .expect("dependencies table");

    if let Some(features) = parsed.get("features").and_then(toml::Value::as_table) {
        assert!(!features.contains_key("semantic-poc"));
    }

    let fastembed = dependencies
        .get("fastembed")
        .and_then(toml::Value::as_table)
        .expect("fastembed dependency");
    assert_ne!(fastembed.get("optional"), Some(&toml::Value::Boolean(true)));
}
