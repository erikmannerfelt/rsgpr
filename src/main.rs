use std::path::Path;

mod dem;
mod io;
mod tools;
mod gpr;

const GPR_ICE_VELOCITY: f32 = 0.168;
const PROGRAM_VERSION: &str = env!("CARGO_PKG_VERSION");
const PROGRAM_NAME: &str = env!("CARGO_PKG_NAME");
const PROGRAM_AUTHORS: &str = env!("CARGO_PKG_AUTHORS");


fn main()-> Result<(), Box<dyn std::error::Error>> {
    let gpr_meta = io::load_rad(Path::new("/media/hdd/Erik/Data/GPR/2022/AG325/glacier_radar/Slakbreen/20220330/DAT_0253_A1.rad"), GPR_ICE_VELOCITY).unwrap();
    //let gpr_meta = load_rad(Path::new("/media/hdd/Erik/Data/GPR/2022/AG325/snow_radar/Slakbreen/20220330/DAT_0053_A1.rad")).unwrap();

    let mut gpr_locations = gpr_meta.find_cor("EPSG:32633").unwrap();
    gpr_locations.get_dem_elevations(Path::new("/media/hdd/Erik/Data/NPI/DEM/NP_S0_DTM20/S0_DTM20.tif"));

    gpr_locations.to_csv(Path::new("locs.csv")).unwrap();
    let mut gpr = gpr::GPR::from_meta_and_loc(gpr_locations, gpr_meta).unwrap();//.subset(Some(0), None, Some(0), None);

    let start = std::time::SystemTime::now();
    println!("Width: {} Height: {}\nt+0ms Running zero-corr", gpr.width(), gpr.height());
    gpr.zero_corr(None);

    println!("t+{:?} Making traces equidistant", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.make_equidistant();
    println!("t+{:?} Normalizing horizontal magnitudes...", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.normalize_horizontal_magnitudes(Some(gpr.height() as isize / 3));
    println!("t+{:?} Migrating", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.kirchoff_migration2d();
    println!("t+{:?} Running dewow", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.dewow(5);

    println!("t+{:?} Normalizing horizontal magnitudes...", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.normalize_horizontal_magnitudes(Some(gpr.height() as isize / 3));

    println!("t+{:?} Automatically finding gain", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.auto_gain(50);

    gpr.export(Path::new("gpr_data.nc"))?;

    println!("t+{:?} Saving ", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.render(&Path::new("img.jpg"))?;

    Ok(())
}





