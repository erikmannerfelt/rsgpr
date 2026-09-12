//! # ridal --- Speeding up Ground Penetrating Radar (GPR) processing
//! A Ground Penetrating Radar (GPR) processing tool written in rust.

mod cli;
mod coords;
mod dem;
mod export;
mod filters;
mod formats;
mod gpr;
mod identity;
mod interp;
mod io;
mod project;
/// Radargram rendering: source window -> view -> resample -> normalize ->
/// colormap -> encode. Outside the `server` feature because nothing in it
/// is about HTTP -- the CLI's `ridal render` and the web GUI are two
/// callers of one pipeline, and `image` was already an unconditional
/// dependency, so this costs a CLI-only build nothing.
mod render;
#[cfg(feature = "server")]
mod server;
/// Windowed NetCDF reads for the renderer above. Moved out of `server`
/// with it: a chunk of a radargram is not a server concept.
mod source;
mod tools;
mod user_metadata;

#[allow(dead_code)]
const PROGRAM_VERSION: &str = env!("CARGO_PKG_VERSION");
#[allow(dead_code)]
const PROGRAM_NAME: &str = env!("CARGO_PKG_NAME");
#[allow(dead_code)]
const PROGRAM_AUTHORS: &str = env!("CARGO_PKG_AUTHORS");

/// Python interface for ridal.
///
/// ridal provides fast, Rust-backed tools for reading, inspecting, and
/// processing ground-penetrating radar (GPR) data from Python.
///
/// Main entry points
/// -----------------
/// read(...)
///     Read one or more GPR files into memory without applying a processing
///     workflow.
/// info(...)
///     Inspect one or more GPR files and return metadata summaries.
/// process(...)
///     Process one or more GPR files into a single output or an in-memory
///     dataset.
/// batch_process(...)
///     Batch-process one or more GPR files into multiple outputs.
///
/// Discovery helpers
/// -----------------
/// all_steps, all_step_descriptions
///     Available processing steps and their descriptions.
/// all_formats, all_format_descriptions
///     Supported file formats and their capabilities.
/// version, __version__
///     Installed ridal version.
///
/// Notes
/// -----
/// `xarray` is an optional dependency. If it is installed, some functions can
/// return `xarray.Dataset` objects. Otherwise, use the plain Python dataset
/// representations such as `"xarray_dict"`.
#[cfg(feature = "python")]
#[pyo3::pymodule]
pub mod ridal {
    use crate::{formats, gpr};
    use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
    use pyo3::ffi::c_str;
    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn optional_metadata(
        py: Python<'_>,
        value: Option<Py<PyAny>>,
    ) -> PyResult<crate::user_metadata::UserMetadata> {
        match value {
            None => Ok(crate::user_metadata::UserMetadata::new()),
            Some(obj) => {
                let bound = obj.bind(py);
                let json = py.import("json")?;
                let text: String = json.getattr("dumps")?.call1((bound,))?.extract()?;
                let value: serde_json::Value = serde_json::from_str(&text)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e:?}")))?;
                crate::user_metadata::value_to_metadata(value)
                    .map_err(pyo3::exceptions::PyValueError::new_err)
            }
        }
    }

    fn json_to_py(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
        let json = py.import("json")?;
        Ok(json.getattr("loads")?.call1((text,))?.unbind())
    }

    fn xarray_dict_to_ds(py: Python<'_>, dict: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let xarray = py.import("xarray")?;

        Ok(xarray
            .getattr("Dataset")?
            .getattr("from_dict")?
            .call1((&dict,))?
            .unbind())
    }

    fn fspath(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
        let os = py.import("os")?;
        let path = os.getattr("fspath")?.call1((value,))?;
        let path_str: String = path.extract()?;

        let os_path = os.getattr("path")?;
        let expanded = os_path.getattr("expanduser")?.call1((path_str,))?;
        let expanded_str: String = expanded.extract()?;

        Ok(PathBuf::from(expanded_str))
    }

    fn inputs_to_paths(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Vec<PathBuf>> {
        if let Ok(list) = value.cast::<PyList>() {
            return list.iter().map(|item| fspath(py, &item)).collect();
        }
        if let Ok(tuple) = value.cast::<PyTuple>() {
            return tuple.iter().map(|item| fspath(py, &item)).collect();
        }
        Ok(vec![fspath(py, value)?])
    }

    fn optional_path(py: Python<'_>, value: Option<Py<PyAny>>) -> PyResult<Option<PathBuf>> {
        match value {
            Some(obj) => {
                let bound = obj.bind(py);
                Ok(Some(fspath(py, bound)?))
            }
            None => Ok(None),
        }
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();
        m.add("version", crate::PROGRAM_VERSION)?;
        m.add("__version__", crate::PROGRAM_VERSION)?;

        let all_steps = gpr::all_available_steps();
        let step_names = all_steps
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<String>>();
        let step_descriptions = all_steps
            .iter()
            .map(|(name, description)| (name.clone(), description.clone()))
            .collect::<BTreeMap<String, String>>();

        m.add("all_steps", step_names)?;
        m.add(
            "all_step_descriptions",
            json_to_py(py, &serde_json::to_string(&step_descriptions).unwrap())?,
        )?;

        let all_formats = formats::all_formats();
        let format_names = all_formats
            .iter()
            .map(|fmt| fmt.name.to_string())
            .collect::<Vec<String>>();

        let format_descriptions = all_formats
            .iter()
            .map(|fmt| {
                (
                    fmt.name.to_string(),
                    serde_json::json!({
                        "description": fmt.description,
                        "capabilities": {
                            "read": fmt.capabilities.read,
                            "write": fmt.capabilities.write,
                        },
                        "files": {
                            "header": fmt.files.header,
                            "data": fmt.files.data,
                            "coordinates": fmt.files.coordinates,
                        }
                    }),
                )
            })
            .collect::<BTreeMap<String, serde_json::Value>>();

        m.add("all_formats", format_names)?;
        m.add(
            "all_format_descriptions",
            json_to_py(py, &serde_json::to_string(&format_descriptions).unwrap())?,
        )?;
        Ok(())
    }

    /// Process one or more GPR files into a single output or an in-memory dataset.
    ///
    /// Use this function when you want to modify or process radar data. One or more
    /// input files are read and optionally corrected or transformed. By default,
    /// the result is written to a single output dataset and the output path is
    /// returned. If `return_dataset=True`, the processed result is returned in
    /// memory instead of being written to disk.
    ///
    /// Parameters
    /// ----------
    /// inputs : path-like or sequence of path-like
    ///     One or more input files to read. A single path, list, or tuple of
    ///     path-like objects is accepted.
    /// output : path-like, optional
    ///     Output file path or output directory. If omitted, a default output path
    ///     is derived from the first input. Ignored when `return_dataset=True`.
    /// steps : str or sequence of str, optional
    ///     Processing steps to apply. This may be given either as a comma-separated
    ///     string or as a sequence of step names.
    ///
    ///     Available steps are exposed as `ridal.all_steps`, and descriptions are
    ///     available in `ridal.all_step_descriptions`.
    ///
    ///     Exactly one of `steps`, `default`, and `default_with_topo` may be given.
    /// return_dataset : bool, default False
    ///     If True, return the processed data as an in-memory dataset object instead
    ///     of writing the dataset to disk. In this mode, `output`, `render`, and
    ///     `track` must not be provided.
    /// default : bool, default False
    ///     Use the default processing profile.
    ///
    ///     Exactly one of `steps`, `default`, and `default_with_topo` may be given.
    /// default_with_topo : bool, default False
    ///     Use the default processing profile and include topographic correction.
    ///
    ///     Exactly one of `steps`, `default`, and `default_with_topo` may be given.
    /// velocity : float, default 0.168
    ///     Propagation velocity in meters per nanosecond.
    /// cor : path-like, optional
    ///     Coordinate file to use instead of any coordinate information implied by
    ///     the input format.
    /// dem : path-like, optional
    ///     Digital elevation model to sample for topographic information.
    /// crs : str, optional
    ///     Coordinate reference system for interpreting or transforming coordinates.
    ///     If omitted, the most appropriate WGS84 UTM zone is used.
    /// track : path-like, optional
    ///     Output path for exported track data. Not allowed when
    ///     `return_dataset=True`.
    /// quiet : bool, default False
    ///     Reduce logging and progress output.
    /// render_profile : str, optional
    ///     Render profile for `render`: a built-in name, or a path to a TOML
    ///     file describing one. Defaults to the built-in `"default"`. Distinct
    ///     from the *processing* profile selected by `default=` and `steps=`.
    /// render_width : int, optional
    ///     Output width in pixels for `render`. Defaults to one pixel per
    ///     trace; a width beyond the trace count is clamped, not upsampled to.
    /// render : path-like, optional
    ///     Output path for a rendered figure. Not allowed when
    ///     `return_dataset=True`.
    /// no_export : bool, default False
    ///     Run processing without writing the main dataset output. Side outputs such
    ///     as rendered figures or exported tracks may still be produced.
    /// override_antenna_mhz : float, optional
    ///     Override the antenna center frequency inferred from the input data.
    /// override_antenna_separation : float, optional
    ///     Override the antenna separation inferred from the input data.
    /// metadata : mapping, optional
    ///     Additional user metadata to attach to the result. This should be a
    ///     JSON-serializable mapping. Root keys are interpreted as strings.
    /// return_dataset_format : str, default "xarray_dict"
    ///     Format used when `return_dataset=True`.
    ///
    ///     Supported values currently include:
    ///
    ///     - ``"xarray_dict"`` for a plain Python representation that does not
    ///       require importing `xarray`.
    ///     - ``"xarray"`` for an `xarray.Dataset`, which requires `xarray` to be
    ///       installed.
    ///
    ///     More return formats may be added in the future.
    ///
    /// Returns
    /// -------
    /// str or object
    ///     The output dataset path as a string in normal export mode, or an
    ///     in-memory dataset object when `return_dataset=True`.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If incompatible arguments are provided, including:
    ///
    ///     - more than one of `steps`, `default`, and `default_with_topo`
    ///     - `return_dataset=True` together with `output`, `render`, or `track`
    ///     - an invalid `steps` value
    /// RuntimeError
    ///     If processing fails.
    /// NotImplementedError
    ///     If `return_dataset_format` is not supported.
    ///
    /// Notes
    /// -----
    /// `process()` is the main processing entry point and is intended for workflows
    /// that modify the data. For lightweight loading of raw data without heavy
    /// processing, use `read()`.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        inputs,
        output=None,
        *,
        steps=None,
        return_dataset=false,
        default=false,
        default_with_topo=false,
        velocity=0.168,
        cor=None,
        dem=None,
        crs=None,
        track=None,
        quiet=false,
        render=None,
        render_profile=None,
        render_width=None,
        no_export=false,
        override_antenna_mhz=None,
        override_antenna_separation=None,
        metadata=None,
        return_dataset_format="xarray_dict".to_string()
    ))]
    fn process(
        py: Python<'_>,
        inputs: Py<PyAny>,
        output: Option<Py<PyAny>>,
        steps: Option<Py<PyAny>>,
        return_dataset: bool,
        default: bool,
        default_with_topo: bool,
        velocity: f32,
        cor: Option<Py<PyAny>>,
        dem: Option<Py<PyAny>>,
        crs: Option<String>,
        track: Option<Py<PyAny>>,
        quiet: bool,
        render: Option<Py<PyAny>>,
        render_profile: Option<String>,
        render_width: Option<usize>,
        no_export: bool,
        override_antenna_mhz: Option<f32>,
        override_antenna_separation: Option<f32>,
        metadata: Option<Py<PyAny>>,
        return_dataset_format: String,
    ) -> PyResult<Py<PyAny>> {
        use pyo3::exceptions::PyValueError;

        if !["xarray", "xarray_dict"]
            .iter()
            .any(|s| s == &return_dataset_format)
        {
            return Err(PyNotImplementedError::new_err(
                "Only 'xarray_dict' and 'xarray' return formats are supported for now",
            ));
        };

        if return_dataset {
            if output.is_some() {
                return Err(PyValueError::new_err(
                    "return_dataset=True requires output=None",
                ));
            }
            if render.is_some() {
                return Err(PyValueError::new_err(
                    "return_dataset=True is incompatible with render=...",
                ));
            }
            if track.is_some() {
                return Err(PyValueError::new_err(
                    "return_dataset=True is incompatible with track=...",
                ));
            }
        }
        let profile_flags =
            usize::from(steps.is_some()) + usize::from(default) + usize::from(default_with_topo);

        if profile_flags > 1 {
            return Err(PyValueError::new_err(
                "Only one of steps=..., default=True, and default_with_topo=True may be provided",
            ));
        }
        let input_paths = inputs_to_paths(py, inputs.bind(py))?;
        let output_path = optional_path(py, output)?;
        let cor_path = optional_path(py, cor)?;
        let dem_path = optional_path(py, dem)?;
        let track_path = match track {
            Some(obj) => Some(Some(fspath(py, obj.bind(py))?)),
            None => None,
        };
        let render_path = match render {
            Some(obj) => Some(Some(fspath(py, obj.bind(py))?)),
            None => None,
        };

        let steps_text = match steps {
            Some(step_obj) => {
                let bound = step_obj.bind(py);
                if let Ok(step_text) = bound.extract::<String>() {
                    Some(step_text)
                } else if let Ok(step_list) = bound.cast::<PyList>() {
                    let parts = step_list
                        .iter()
                        .map(|item| item.extract::<String>())
                        .collect::<PyResult<Vec<String>>>()?;
                    Some(parts.join(","))
                } else if let Ok(step_tuple) = bound.cast::<PyTuple>() {
                    let parts = step_tuple
                        .iter()
                        .map(|item| item.extract::<String>())
                        .collect::<PyResult<Vec<String>>>()?;
                    Some(parts.join(","))
                } else {
                    return Err(PyValueError::new_err(
                        "steps must be a string or a list/tuple of strings",
                    ));
                }
            }
            None => None,
        };

        let resolved_steps = if default_with_topo {
            let mut profile = gpr::default_processing_profile();
            profile.push("correct_topography".to_string());
            profile
        } else if default {
            gpr::default_processing_profile()
        } else if let Some(step_text) = steps_text.as_deref() {
            crate::tools::parse_step_list(step_text).map_err(PyValueError::new_err)?
        } else {
            vec![]
        };

        gpr::validate_steps(&resolved_steps).map_err(PyValueError::new_err)?;

        let user_metadata = optional_metadata(py, metadata)?;

        if return_dataset {
            // Build but do not export
            let params2 = gpr::RunParams {
                filepaths: input_paths,
                output_path: None,
                dem_path,
                cor_path,
                medium_velocity: velocity,
                crs,
                quiet,
                track_path: None,
                steps: resolved_steps,
                no_export: true,
                render_path: None,
                render_profile: None,
                render_width: None,
                override_antenna_mhz,
                override_antenna_separation,
                user_metadata,
                // Not yet exposed as Python kwargs (#116); falls back to the
                // output-stem-derived radargram ID, same as the CLI default.
                radargram_id: None,
                display_name: None,
                group: None,
                group_id: None,
            };
            let (gpr_obj, _default_path) = gpr::build_processed_gpr(params2)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            let ds = gpr_obj
                .export_dataset()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            let ds_py = ds.to_python(py);
            if return_dataset_format == "xarray_dict" {
                return ds_py;
            } else if return_dataset_format == "xarray" {
                return xarray_dict_to_ds(py, ds_py?);
            } else {
                unreachable!()
            }
        }

        // file/export mode (unchanged)
        let params = gpr::RunParams {
            filepaths: input_paths,
            output_path,
            dem_path,
            cor_path,
            medium_velocity: velocity,
            crs,
            quiet,
            track_path,
            steps: resolved_steps,
            no_export,
            render_path,
            render_profile,
            render_width,
            override_antenna_mhz,
            override_antenna_separation,
            user_metadata,
            radargram_id: None,
            display_name: None,
            group: None,
            group_id: None,
        };
        let result = gpr::run(params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        // Ok(result.output_path.to_string_lossy().to_string())
        Ok(py
            .eval(c_str!("str"), None, None)?
            .call1((result.output_path.to_string_lossy().to_string(),))?
            .unbind())
    }
    /// Read one or more GPR files into memory without applying a processing workflow.
    ///
    /// Use this function to load radar data in a lightweight form for inspection,
    /// exploration, or downstream processing in Python. Unlike `process()`,
    /// `read()` is intended to return the data essentially as read from disk rather
    /// than applying a full processing workflow.
    ///
    /// Parameters
    /// ----------
    /// inputs : path-like or sequence of path-like
    ///     One or more input files to read. A single path, list, or tuple of
    ///     path-like objects is accepted.
    /// velocity : float, default 0.168
    ///     Propagation velocity in meters per nanosecond.
    /// cor : path-like, optional
    ///     Coordinate file to use instead of any coordinate information implied by
    ///     the input format.
    /// dem : path-like, optional
    ///     Digital elevation model to sample for topographic information.
    /// crs : str, optional
    ///     Coordinate reference system for interpreting or transforming coordinates.
    ///     If omitted, the most appropriate WGS84 UTM zone is used.
    /// override_antenna_mhz : float, optional
    ///     Override the antenna center frequency inferred from the input data.
    /// metadata : mapping, optional
    ///     Additional user metadata to attach to the returned dataset. This should
    ///     be a JSON-serializable mapping. Root keys are interpreted as strings.
    /// return_dataset_format : str, default "xarray_dict"
    ///     Format of the returned in-memory dataset.
    ///
    ///     Supported values currently include:
    ///
    ///     - ``"xarray_dict"`` for a plain Python representation that does not
    ///       require importing `xarray`.
    ///     - ``"xarray"`` for an `xarray.Dataset`, which requires `xarray` to be
    ///       installed.
    ///
    ///     More return formats may be added in the future.
    ///
    /// Returns
    /// -------
    /// object
    ///     An in-memory dataset representation of the input data.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If reading fails.
    /// NotImplementedError
    ///     If `return_dataset_format` is not supported.
    ///
    /// Notes
    /// -----
    /// `read()` is intended as a lightweight loader. If you want to apply filtering,
    /// corrections, or export a processed dataset, use `process()` instead.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        inputs,
        *,
        velocity=0.168,
        cor=None,
        dem=None,
        crs=None,
        override_antenna_mhz=None,
        override_antenna_separation=None,
        metadata=None,
        return_dataset_format="xarray_dict".to_string()
    ))]
    fn read(
        py: Python<'_>,
        inputs: Py<PyAny>,
        velocity: f32,
        cor: Option<Py<PyAny>>,
        dem: Option<Py<PyAny>>,
        crs: Option<String>,
        override_antenna_mhz: Option<f32>,
        override_antenna_separation: Option<f32>,
        metadata: Option<Py<PyAny>>,
        return_dataset_format: String,
    ) -> PyResult<Py<PyAny>> {
        process(
            py,
            inputs,
            None,
            None,
            true,
            false,
            false,
            velocity,
            cor,
            dem,
            crs,
            None,
            true,
            None,
            None,
            None,
            false,
            override_antenna_mhz,
            override_antenna_separation,
            metadata,
            return_dataset_format,
        )
    }
    /// Batch-process one or more GPR files into multiple outputs.
    ///
    /// Use this function when many input files should be processed in one call and
    /// written as separate outputs in an existing output directory.
    ///
    /// Parameters
    /// ----------
    /// inputs : path-like or sequence of path-like
    ///     One or more input files to process. A single path, list, or tuple of
    ///     path-like objects is accepted.
    /// output : path-like
    ///     Existing output directory where processed datasets will be written.
    /// steps : str or sequence of str, optional
    ///     Processing steps to apply. This may be given either as a comma-separated
    ///     string or as a sequence of step names.
    ///
    ///     Available steps are exposed as `ridal.all_steps`, and descriptions are
    ///     available in `ridal.all_step_descriptions`.
    ///
    ///     Exactly one of `steps`, `default`, and `default_with_topo` may be given.
    /// default : bool, default False
    ///     Use the default processing profile.
    ///
    ///     Exactly one of `steps`, `default`, and `default_with_topo` may be given.
    /// default_with_topo : bool, default False
    ///     Use the default processing profile and include topographic correction.
    ///
    ///     Exactly one of `steps`, `default`, and `default_with_topo` may be given.
    /// velocity : float, default 0.168
    ///     Propagation velocity in meters per nanosecond.
    /// cor : path-like, optional
    ///     Coordinate file to use instead of any coordinate information implied by
    ///     the input format.
    /// dem : path-like, optional
    ///     Digital elevation model to sample for topographic information.
    /// crs : str, optional
    ///     Coordinate reference system for interpreting or transforming coordinates.
    ///     If omitted, the most appropriate WGS84 UTM zone is used.
    /// track : path-like, optional
    ///     Existing directory where exported track files should be written.
    /// quiet : bool, default False
    ///     Reduce logging and progress output.
    /// render : path-like, optional
    ///     Existing directory where rendered figures should be written.
    /// render_profile : str, optional
    ///     Render profile for `render`: a built-in name, or a path to a TOML
    ///     file describing one. Defaults to the built-in `"default"`. Distinct
    ///     from the *processing* profile selected by `default=` and `steps=`.
    /// render_width : int, optional
    ///     Output width in pixels for `render`. Defaults to one pixel per
    ///     trace; a width beyond the trace count is clamped, not upsampled to.
    /// no_export : bool, default False
    ///     Run processing without writing the main dataset outputs. Side outputs
    ///     such as rendered figures or exported tracks may still be produced.
    /// merge : str, optional
    ///     Merge chronologically neighboring profiles when they are close enough in
    ///     time and otherwise compatible.
    ///
    ///     For example, ``"10 min"`` will merge neighboring profiles separated by
    ///     less than ten minutes.
    ///
    ///     The value is parsed using the `parse_duration` syntax. Briefly, it
    ///     accepts sequences of ``[value] [unit]`` pairs such as
    ///     ``"15 days 20 seconds 100 milliseconds"``; spaces are optional, and
    ///     unit order does not matter. See the full syntax and accepted
    ///     abbreviations at:
    ///     https://docs.rs/parse_duration/latest/parse_duration/#syntax
    /// override_antenna_mhz : float, optional
    ///     Override the antenna center frequency inferred from the input data.
    /// metadata : mapping, optional
    ///     Additional user metadata to attach independently to each produced output.
    ///     This should be a JSON-serializable mapping. Root keys are interpreted as
    ///     strings.
    ///
    /// Returns
    /// -------
    /// list of str
    ///     Output dataset paths as strings, in the order produced.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If incompatible arguments are provided, including:
    ///
    ///     - more than one of `steps`, `default`, and `default_with_topo`
    ///     - `output` is not an existing directory
    ///     - `track` is provided but is not an existing directory
    ///     - `render` is provided but is not an existing directory
    ///     - an invalid `steps` value
    /// RuntimeError
    ///     If batch processing fails.
    ///
    /// Notes
    /// -----
    /// Unlike `process()`, `batch_process()` always targets an existing output
    /// directory and produces multiple outputs.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
    inputs,
    output,
    *,
    steps=None,
    default=false,
    default_with_topo=false,
    velocity=0.168,
    cor=None,
    dem=None,
    crs=None,
    track=None,
    quiet=false,
    render=None,
    render_profile=None,
    render_width=None,
        no_export=false,
        merge=None,
        override_antenna_mhz=None,
        override_antenna_separation=None,
        metadata=None,
 ))]
    fn batch_process(
        py: Python<'_>,
        inputs: Py<PyAny>,
        output: Py<PyAny>,
        steps: Option<Py<PyAny>>,
        default: bool,
        default_with_topo: bool,
        velocity: f32,
        cor: Option<Py<PyAny>>,
        dem: Option<Py<PyAny>>,
        crs: Option<String>,
        track: Option<Py<PyAny>>,
        quiet: bool,
        render: Option<Py<PyAny>>,
        render_profile: Option<String>,
        render_width: Option<usize>,
        no_export: bool,
        merge: Option<String>,
        override_antenna_mhz: Option<f32>,
        override_antenna_separation: Option<f32>,
        metadata: Option<Py<PyAny>>,
    ) -> PyResult<Vec<String>> {
        use pyo3::exceptions::{PyRuntimeError, PyValueError};

        let input_paths = inputs_to_paths(py, inputs.bind(py))?;
        let output_dir = fspath(py, output.bind(py))?;
        if !output_dir.is_dir() {
            return Err(PyValueError::new_err(format!(
                "output must be an existing directory in batch_process(): {}",
                output_dir.display()
            )));
        }

        let cor_path = optional_path(py, cor)?;
        let dem_path = optional_path(py, dem)?;
        let track_dir = match track {
            Some(obj) => {
                let p = fspath(py, obj.bind(py))?;
                if !p.is_dir() {
                    return Err(PyValueError::new_err(format!(
                        "track must be an existing directory in batch_process(): {}",
                        p.display()
                    )));
                }
                Some(p)
            }
            None => None,
        };
        let render_dir = match render {
            Some(obj) => {
                let p = fspath(py, obj.bind(py))?;
                if !p.is_dir() {
                    return Err(PyValueError::new_err(format!(
                        "render must be an existing directory in batch_process(): {}",
                        p.display()
                    )));
                }
                Some(p)
            }
            None => None,
        };

        let profile_flags =
            usize::from(steps.is_some()) + usize::from(default) + usize::from(default_with_topo);

        if profile_flags > 1 {
            return Err(PyValueError::new_err(
                "Only one of steps=..., default=True, and default_with_topo=True may be provided",
            ));
        }
        let steps_text = match steps {
            Some(step_obj) => {
                let bound = step_obj.bind(py);
                if let Ok(step_text) = bound.extract::<String>() {
                    Some(step_text)
                } else if let Ok(step_list) = bound.cast::<PyList>() {
                    let parts = step_list
                        .iter()
                        .map(|item| item.extract::<String>())
                        .collect::<PyResult<Vec<String>>>()?;
                    Some(parts.join(","))
                } else if let Ok(step_tuple) = bound.cast::<PyTuple>() {
                    let parts = step_tuple
                        .iter()
                        .map(|item| item.extract::<String>())
                        .collect::<PyResult<Vec<String>>>()?;
                    Some(parts.join(","))
                } else {
                    return Err(PyValueError::new_err(
                        "steps must be a string or a list/tuple of strings",
                    ));
                }
            }
            None => None,
        };

        let resolved_steps = if default_with_topo {
            let mut profile = gpr::default_processing_profile();
            profile.push("correct_topography".to_string());
            profile
        } else if default {
            gpr::default_processing_profile()
        } else if let Some(step_text) = steps_text.as_deref() {
            crate::tools::parse_step_list(step_text).map_err(PyValueError::new_err)?
        } else {
            vec![]
        };

        gpr::validate_steps(&resolved_steps).map_err(PyValueError::new_err)?;
        let user_metadata = optional_metadata(py, metadata)?;

        let params = gpr::BatchRunParams {
            filepaths: input_paths,
            output_dir,
            dem_path,
            cor_path,
            medium_velocity: velocity,
            crs,
            quiet,
            track_dir,
            steps: resolved_steps,
            no_export,
            render_dir,
            render_profile,
            render_width,
            merge,
            override_antenna_mhz,
            override_antenna_separation,
            user_metadata,
            group: None,
            group_id: None,
        };

        let result =
            gpr::run_batch(params).map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;

        Ok(result
            .output_paths
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect())
    }

    /// Inspect one or more GPR files and return metadata summaries.
    ///
    /// This function reads metadata and summary information without performing a
    /// full processing workflow.
    ///
    /// Parameters
    /// ----------
    /// inputs : path-like or sequence of path-like
    ///     One or more input files to inspect. A single path, list, or tuple of
    ///     path-like objects is accepted.
    /// velocity : float, default 0.168
    ///     Propagation velocity in meters per nanosecond.
    /// cor : path-like, optional
    ///     Coordinate file to use instead of any coordinate information implied by
    ///     the input format.
    /// dem : path-like, optional
    ///     Digital elevation model to sample for topographic information.
    /// crs : str, optional
    ///     Coordinate reference system for interpreting or transforming coordinates.
    ///     If omitted, the most appropriate WGS84 UTM zone is used.
    /// override_antenna_mhz : float, optional
    ///     Override the antenna center frequency inferred from the input data.
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     One metadata summary dictionary per input file.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If inspection fails.
    ///
    /// Render a processed radargram to an image file.
    ///
    /// The Python equivalent of `ridal render`, and the same renderer the web
    /// GUI uses, so an image produced here matches one downloaded from the
    /// browser for the same file and profile.
    ///
    /// Parameters
    /// ----------
    /// input : path-like
    ///     Processed `.nc` file to render.
    /// output : path-like, optional
    ///     Output image path. The extension selects the encoding (`.png` or
    ///     `.jpg`). If omitted, a sidecar beside the input is written, using
    ///     the profile's own format.
    /// profile : str, default "default"
    ///     A built-in render profile name, or a path to a TOML file describing
    ///     one. Built-in names win, so a file of the same name in the working
    ///     directory cannot shadow them.
    /// width : int, optional
    ///     Output width in pixels. Defaults to one pixel per trace. A width
    ///     beyond the trace count is clamped rather than upsampled to.
    /// quality : int, optional
    ///     JPEG quality, 1-100. Ignored when the output is PNG.
    ///
    /// Returns
    /// -------
    /// dict
    ///     `{"path": str, "width": int, "height": int, "profile": str}`.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the input cannot be read, the profile cannot be resolved, or the
    ///     image cannot be written.
    ///
    /// Notes
    /// -----
    /// To render while processing raw data instead, pass `render=` to
    /// `process()` or `batch_process()`; those accept the same `render_profile`
    /// and `render_width` and go through this same pipeline.
    #[pyfunction]
    #[pyo3(signature = (input, output=None, *, profile=None, width=None, quality=None))]
    fn render(
        py: Python<'_>,
        input: Py<PyAny>,
        output: Option<Py<PyAny>>,
        profile: Option<String>,
        width: Option<usize>,
        quality: Option<u8>,
    ) -> PyResult<Py<PyAny>> {
        use pyo3::exceptions::PyRuntimeError;

        let input = fspath(py, input.bind(py))?;
        let resolved =
            crate::render::profile::RenderProfile::resolve(profile.as_deref().unwrap_or("default"))
                .map_err(PyRuntimeError::new_err)?;
        let output = match output {
            Some(value) => fspath(py, value.bind(py))?,
            None => crate::render::oneshot::sidecar_path(&input, &resolved),
        };

        let request = crate::render::oneshot::RenderRequest {
            profile: &resolved,
            width,
            quality,
        };
        let (w, h) = crate::render::oneshot::render_path_to_file(&input, &output, &request)
            .map_err(PyRuntimeError::new_err)?;

        let out = pyo3::types::PyDict::new(py);
        out.set_item("path", output.display().to_string())?;
        out.set_item("width", w)?;
        out.set_item("height", h)?;
        out.set_item("profile", resolved.name)?;
        Ok(out.into_any().unbind())
    }

    /// Notes
    /// -----
    /// `info()` is intended for lightweight inspection. For loading in-memory data,
    /// use `read()`. For modifying or exporting processed data, use `process()` or
    /// `batch_process()`.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        inputs,
        *,
        velocity=0.168,
        cor=None,
        dem=None,
        crs=None,
        override_antenna_mhz=None,
        override_antenna_separation=None,
    ))]
    fn info(
        py: Python<'_>,
        inputs: Py<PyAny>,
        velocity: f32,
        cor: Option<Py<PyAny>>,
        dem: Option<Py<PyAny>>,
        crs: Option<String>,
        override_antenna_mhz: Option<f32>,
        override_antenna_separation: Option<f32>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let input_paths = inputs_to_paths(py, inputs.bind(py))?;
        let cor_path = optional_path(py, cor)?;
        let dem_path = optional_path(py, dem)?;

        let params = gpr::InfoParams {
            filepaths: input_paths,
            dem_path,
            cor_path,
            medium_velocity: velocity,
            crs,
            override_antenna_mhz,
            override_antenna_separation,
        };
        let records =
            gpr::inspect(params).map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;

        records
            .iter()
            .map(|r| {
                let text = serde_json::to_string(r).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to serialize info record to JSON: {e}"))
                })?;
                json_to_py(py, &text)
            })
            .collect::<PyResult<Vec<Py<PyAny>>>>()
    }

    /// Removed legacy entry point for the old Python CLI wrapper.
    ///
    /// `ridal.run_cli()` is no longer supported. Use `ridal.process()` for
    /// processing workflows and `ridal.info()` for metadata inspection.
    ///
    /// Raises
    /// ------
    /// NotImplementedError
    ///     Always raised.
    #[pyfunction(signature = (*_args, **_kwargs))]
    fn run_cli(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "ridal.run_cli() has been removed. Use ridal.process(...) for processing and ridal.info(...) for metadata inspection.",
        ))
    }
}
