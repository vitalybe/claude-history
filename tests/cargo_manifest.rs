use std::{fs, path::Path};

#[test]
fn default_features_do_not_enable_fastembed() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest = fs::read_to_string(manifest_path).expect("read Cargo.toml");
    let parsed: toml::Value = toml::from_str(&manifest).expect("parse Cargo.toml");
    let features = parsed
        .get("features")
        .and_then(toml::Value::as_table)
        .expect("features table");
    let dependencies = parsed
        .get("dependencies")
        .and_then(toml::Value::as_table)
        .expect("dependencies table");

    let default_features = features
        .get("default")
        .and_then(toml::Value::as_array)
        .expect("default feature list");
    assert!(default_features.is_empty());

    let semantic_poc = features
        .get("semantic-poc")
        .and_then(toml::Value::as_array)
        .expect("semantic-poc feature list");
    assert_eq!(
        semantic_poc,
        &vec![toml::Value::String("dep:fastembed".to_string())]
    );

    let fastembed = dependencies
        .get("fastembed")
        .and_then(toml::Value::as_table)
        .expect("fastembed dependency");
    assert_eq!(fastembed.get("optional"), Some(&toml::Value::Boolean(true)));
}
