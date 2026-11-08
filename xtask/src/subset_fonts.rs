//! Regenerate the frontend's Iosevka Extended web fonts from a pinned
//! upstream release.
//!
//! The outputs under `frontend/src/assets/fonts/` are committed and are the
//! source of truth at build time; this command only exists to regenerate
//! them. It takes local copies of two upstream files, checks each against a
//! pinned SHA-256, and writes:
//!
//! - `iosevka-extended-{400,500,700}.woff2`, subset to [`UNICODE_RANGES`]
//!   with HarfBuzz's default layout features plus `zero`, and hinting
//!   dropped,
//! - `iosevka.css`, the matching `@font-face` rules, so the CSS
//!   `unicode-range` and the subset cannot drift apart,
//! - `Iosevka-LICENSE.md`, the upstream licence, unchanged, and
//! - `README.md`, recording the provenance below.
//!
//! The inputs, for release [`RELEASE`]:
//!
//! ```text
//! https://github.com/be5invis/Iosevka/releases/download/v34.9.0/PkgTTF-Iosevka-34.9.0.zip
//! https://raw.githubusercontent.com/be5invis/Iosevka/v34.9.0/LICENSE.md
//! ```
//!
//! ```sh
//! cargo xtask subset-fonts --archive PkgTTF-Iosevka-34.9.0.zip --license LICENSE.md
//! ```
//!
//! To move to a newer release, update [`RELEASE`] and both digests together;
//! a digest that does not match is an error, not a warning.

use std::{
    fmt, fs,
    io::{Cursor, Read as _},
    path::{Path, PathBuf},
};

use clap::Args;
use hb_subset::{Blob, FontFace, SubsetInput, Tag};
use sha2::{Digest as _, Sha256};

#[derive(Args)]
pub struct SubsetFontsArgs {
    /// Path to `PkgTTF-Iosevka-<RELEASE>.zip` from the upstream release.
    #[arg(long)]
    pub archive: PathBuf,
    /// Path to `LICENSE.md` from the upstream tag.
    #[arg(long)]
    pub license: PathBuf,
}

pub fn run(args: SubsetFontsArgs) -> Result<(), Error> {
    let archive = read_verified(&args.archive, ARCHIVE_SHA256)?;
    let license = read_verified(&args.license, LICENSE_SHA256)?;

    let out_dir = repo_root()?.join(OUTPUT_DIR);
    fs::create_dir_all(&out_dir).map_err(|source| Error::Io {
        path: out_dir.clone(),
        source,
    })?;

    let mut zip = zip::ZipArchive::new(Cursor::new(archive)).map_err(|source| Error::Zip {
        entry: None,
        source,
    })?;
    for face in FACES {
        let ttf = extract(&mut zip, face.source)?;
        let subset = subset(&ttf).map_err(|cause| Error::Subset {
            face: face.source,
            cause,
        })?;
        let woff2 = ttf2woff2::encode(&subset, ttf2woff2::BrotliQuality::from(WOFF2_QUALITY))
            .map_err(|source| Error::Woff2 {
                face: face.source,
                source,
            })?;
        let path = out_dir.join(face.output_name());
        write(&path, &woff2)?;
        println!(
            "{}: {} KiB from {} KiB",
            path.display(),
            woff2.len() / 1024,
            ttf.len() / 1024
        );
    }

    write(&out_dir.join("Iosevka-LICENSE.md"), &license)?;
    write(&out_dir.join("iosevka.css"), font_face_css().as_bytes())?;
    write(&out_dir.join("README.md"), readme().as_bytes())?;
    Ok(())
}

// ── pinned inputs ───────────────────────────────────────────────────────

/// The upstream Iosevka release the committed fonts are cut from.
const RELEASE: &str = "34.9.0";

/// SHA-256 of `PkgTTF-Iosevka-34.9.0.zip`, as listed on the release page.
const ARCHIVE_SHA256: &str = "c61a10b880bf3087f982217423fa2876a93803bd2a1901fbc52ab80c7329f716";

/// SHA-256 of `LICENSE.md` at tag `v34.9.0`.
const LICENSE_SHA256: &str = "4ba53c7c1cb39279aae5f8d7d22054c485c71169920e5a36ed098b115e2e3c5d";

/// Output directory, relative to the repository root. Under `src/assets`
/// rather than `public/` so Vite fingerprints the files.
const OUTPUT_DIR: &str = "frontend/src/assets/fonts";

struct Face {
    /// File name inside the release archive.
    source: &'static str,
    weight: u16,
    /// The CSS `font-weight` descriptor; the bold face also answers for
    /// semibold, so 600 maps to it rather than being synthesised.
    css_weight: &'static str,
}

impl Face {
    fn output_name(&self) -> String {
        format!("iosevka-extended-{}.woff2", self.weight)
    }
}

const FACES: &[Face] = &[
    Face {
        source: "Iosevka-Extended.ttf",
        weight: 400,
        css_weight: "400",
    },
    Face {
        source: "Iosevka-ExtendedMedium.ttf",
        weight: 500,
        css_weight: "500",
    },
    Face {
        source: "Iosevka-ExtendedBold.ttf",
        weight: 700,
        css_weight: "600 700",
    },
];

/// Codepoints kept, inclusive. Latin-1 and Latin Extended-A, the spacing
/// modifiers Latin text uses, general punctuation, the euro and trademark
/// signs, arrows, maths operators, miscellaneous technical, box drawing,
/// block elements, and geometric shapes (llama.cpp's progress output), check
/// marks, and the replacement character. Anything else in child logs falls
/// back to the next family in `--font-mono`.
const UNICODE_RANGES: &[(u32, u32)] = &[
    (0x0000, 0x00FF),
    (0x0100, 0x017F),
    (0x02C6, 0x02C6),
    (0x02DA, 0x02DA),
    (0x02DC, 0x02DC),
    (0x2000, 0x206F),
    (0x20AC, 0x20AC),
    (0x2122, 0x2122),
    (0x2190, 0x23FF),
    (0x2500, 0x25FF),
    (0x2713, 0x2717),
    (0xFFFD, 0xFFFD),
];

/// Layout features kept beyond HarfBuzz's default shaping set (which already
/// has `calt` and `liga`): the slashed zero that `--font-mono` turns on.
/// Keeping every feature instead pulls in Iosevka's character-variant
/// alternates and triples the size of each file.
const EXTRA_FEATURES: &[&[u8; 4]] = &[b"zero"];

/// Brotli's best; this runs rarely, so the time is a one-off.
const WOFF2_QUALITY: u8 = 11;

// ── pipeline ────────────────────────────────────────────────────────────

fn read_verified(path: &Path, expected: &'static str) -> Result<Vec<u8>, Error> {
    let bytes = fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let actual = hex(&Sha256::digest(&bytes));
    if actual != expected {
        return Err(Error::Digest {
            path: path.to_path_buf(),
            expected,
            actual,
        });
    }
    Ok(bytes)
}

fn extract(
    zip: &mut zip::ZipArchive<Cursor<Vec<u8>>>,
    name: &'static str,
) -> Result<Vec<u8>, Error> {
    let zip_error = |source| Error::Zip {
        entry: Some(name),
        source,
    };
    let mut entry = zip.by_name(name).map_err(zip_error)?;
    let mut bytes = Vec::new();
    entry
        .read_to_end(&mut bytes)
        .map_err(|source| zip_error(source.into()))?;
    Ok(bytes)
}

fn subset(ttf: &[u8]) -> Result<Vec<u8>, &'static str> {
    let blob = Blob::from_bytes(ttf).map_err(|_| "failed to allocate a blob")?;
    let font = FontFace::new(blob).map_err(|_| "failed to read the font face")?;
    let mut input = SubsetInput::new().map_err(|_| "failed to allocate a subset input")?;
    {
        let mut unicodes = input.unicode_set();
        for &(start, end) in UNICODE_RANGES {
            for c in (start..=end).filter_map(char::from_u32) {
                unicodes.insert(c);
            }
        }
    }
    {
        let mut features = input.layout_feature_tag_set();
        for tag in EXTRA_FEATURES {
            features.insert(Tag::new(*tag));
        }
    }
    input.flags().remove_hinting();
    let subset = input
        .subset_font(&font)
        .map_err(|_| "HarfBuzz failed to subset the font")?;
    Ok(subset.underlying_blob().to_vec())
}

fn font_face_css() -> String {
    let ranges = UNICODE_RANGES
        .iter()
        .map(|&(start, end)| {
            if start == end {
                format!("U+{start:04X}")
            } else {
                format!("U+{start:04X}-{end:04X}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let mut css = format!(
        "/* Generated by `cargo xtask subset-fonts` from Iosevka {RELEASE}; do not edit. */\n"
    );
    for face in FACES {
        css.push_str(&format!(
            "\n@font-face {{\n  font-family: \"Iosevka Extended\";\n  src: url(\"./{}\") format(\"woff2\");\n  font-weight: {};\n  font-style: normal;\n  font-display: swap;\n  unicode-range: {ranges};\n}}\n",
            face.output_name(),
            face.css_weight,
        ));
    }
    css
}

fn readme() -> String {
    format!(
        "# Iosevka Extended\n\n\
         These files are generated by `cargo xtask subset-fonts`; do not edit them by hand. They are subset from Iosevka {RELEASE} (SIL Open Font License 1.1, see `Iosevka-LICENSE.md`; the licence declares no Reserved Font Name, so the subset keeps the name).\n\n\
         Inputs, checked against SHA-256 digests pinned in `xtask/src/subset_fonts.rs`:\n\n\
         - https://github.com/be5invis/Iosevka/releases/download/v{RELEASE}/PkgTTF-Iosevka-{RELEASE}.zip\n\
         - https://raw.githubusercontent.com/be5invis/Iosevka/v{RELEASE}/LICENSE.md\n\n\
         To regenerate:\n\n\
         ```sh\n\
         cargo xtask subset-fonts --archive PkgTTF-Iosevka-{RELEASE}.zip --license LICENSE.md\n\
         ```\n"
    )
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn repo_root() -> Result<PathBuf, Error> {
    let metadata = cargo_metadata::MetadataCommand::new()
        .no_deps()
        .exec()
        .map_err(Error::CargoMetadata)?;
    Ok(metadata.workspace_root.as_std_path().to_path_buf())
}

fn write(path: &Path, contents: &[u8]) -> Result<(), Error> {
    fs::write(path, contents).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

// ── error type ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum Error {
    CargoMetadata(cargo_metadata::Error),
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Digest {
        path: PathBuf,
        expected: &'static str,
        actual: String,
    },
    Zip {
        entry: Option<&'static str>,
        source: zip::result::ZipError,
    },
    Subset {
        face: &'static str,
        cause: &'static str,
    },
    Woff2 {
        face: &'static str,
        source: ttf2woff2::Error,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CargoMetadata(source) => write!(f, "failed to read cargo metadata: {source}"),
            Self::Io { path, source } => {
                write!(f, "failed to access {}: {source}", path.display())
            }
            Self::Digest {
                path,
                expected,
                actual,
            } => write!(
                f,
                "failed to verify {}: SHA-256 is {actual}, expected {expected} (Iosevka {RELEASE})",
                path.display()
            ),
            Self::Zip {
                entry: Some(entry),
                source,
            } => write!(f, "failed to extract {entry} from the archive: {source}"),
            Self::Zip {
                entry: None,
                source,
            } => write!(f, "failed to open the archive: {source}"),
            Self::Subset { face, cause } => write!(f, "failed to subset {face}: {cause}"),
            Self::Woff2 { face, source } => {
                write!(f, "failed to encode {face} as WOFF2: {source}")
            }
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_ranges_are_ascending_and_disjoint() {
        for &(start, end) in UNICODE_RANGES {
            assert!(start <= end, "U+{start:04X}-{end:04X} is reversed");
        }
        for pair in UNICODE_RANGES.windows(2) {
            assert!(
                pair[0].1 < pair[1].0,
                "U+{:04X}-{:04X} overlaps or precedes the range before it",
                pair[1].0,
                pair[1].1
            );
        }
    }

    #[test]
    fn css_declares_every_face_with_the_shared_ranges() {
        let css = font_face_css();
        for face in FACES {
            assert!(css.contains(&face.output_name()));
            assert!(css.contains(&format!("font-weight: {};", face.css_weight)));
        }
        assert_eq!(
            css.matches("unicode-range: U+0000-00FF, U+0100-017F,")
                .count(),
            FACES.len()
        );
    }
}
