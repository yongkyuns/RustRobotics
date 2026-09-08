"""One-off, guarded source transformation; remove before merge."""
from pathlib import Path
import hashlib
import re

root = Path(__file__).resolve().parents[1]

def blob(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()

p = root / 'rust_robotics_algo/src/slam/ekf_slam.rs'
assert blob(p.read_bytes()) == 'd6dfd77c4e90702ba131b46ba288ad92f770f802'
original_ekf = p.read_text()
s = original_ekf
s = s.replace('fn rand() -> f32 {\n    2.0 * (rand::random::<f32>() - 0.5)\n}', 'fn rand<R: rand::Rng + ?Sized>(rng: &mut R) -> f32 {\n    2.0 * (rng.gen::<f32>() - 0.5)\n}')
sig = '''pub fn generate_observations(
    robot_pose: &Vector3<f32>,
    true_landmarks: &[Vector2<f32>],
    config: &EkfSlamConfig,
    add_noise: bool,
) -> Vec<(usize, Observation)> {'''
assert s.count(sig) == 1
s = s.replace(sig, sig + '''
    generate_observations_with_rng(robot_pose, true_landmarks, config, add_noise, &mut rand::thread_rng())
}

// Keep random state caller-owned in tests; the public demo API remains entropy-backed.
fn generate_observations_with_rng<R: rand::Rng + ?Sized>(
    robot_pose: &Vector3<f32>,
    true_landmarks: &[Vector2<f32>],
    config: &EkfSlamConfig,
    add_noise: bool,
    rng: &mut R,
) -> Vec<(usize, Observation)> {''')
pre, tests = s.split('#[cfg(test)]\nmod tests {', 1)
pre = pre.replace('let range_noise = rand()', 'let range_noise = rand(rng)').replace('let bearing_noise = rand()', 'let bearing_noise = rand(rng)')
step = '    // 1. Prediction step\n    predict(state, config, v, w, dt);'
assert pre.count(step) == 1
pre = pre.replace(step, '''    step_with_rng(state, config, true_pose, true_landmarks, (v, w, dt), &mut rand::thread_rng());
}

fn step_with_rng<R: rand::Rng + ?Sized>(
    state: &mut EkfSlamState,
    config: &EkfSlamConfig,
    true_pose: &Vector3<f32>,
    true_landmarks: &[Vector2<f32>],
    control: (f32, f32, f32),
    rng: &mut R,
) {
    let (v, w, dt) = control;
''' + step)
pre = pre.replace('generate_observations(true_pose, true_landmarks, config, true);', 'generate_observations_with_rng(true_pose, true_landmarks, config, true, rng);')
tests = tests.replace('    use super::*;', '    use super::*;\n    use rand::{rngs::StdRng, Rng, SeedableRng};', 1)

def add_last_arg(text, name, new_name, argument):
    pattern = re.compile(r'\b' + name + r'\(')
    for match in reversed(list(pattern.finditer(text))):
        begin = match.end()
        level, i = 1, begin
        while level:
            if text[i] == '(':
                level += 1
            elif text[i] == ')':
                level -= 1
            i += 1
        args = text[begin:i - 1].rstrip().rstrip(',')
        text = text[:match.start()] + new_name + '(' + args + ', ' + argument + ')' + text[i:]
    return text

seeded = []
for index, match in reversed(list(enumerate(re.finditer(r'^    fn (test_\w+)\(\) \{', tests, re.M)))):
    end = tests.index('\n    }', match.end())
    body = tests[match.end():end]
    if not re.search(r'\b(?:rand|generate_observations|generate_random_landmarks_ring|step)\(|rand::random', body):
        continue
    seed = 0xE4F00000 + index
    seeded.append((match.group(1), seed))
    body = body.replace('rand()', 'rand(&mut rng)').replace('rand::random::<f32>()', 'rng.gen::<f32>()')
    body = add_last_arg(body, 'generate_observations', 'generate_observations_with_rng', '&mut rng')
    body = add_last_arg(body, 'generate_random_landmarks_ring', 'generate_random_landmarks_ring', '&mut rng')
    body = body.replace('step(&mut state, &config, &true_pose, &landmarks, v, w, dt);', 'step_with_rng(&mut state, &config, &true_pose, &landmarks, (v, w, dt), &mut rng);')
    body = f'\n        let mut rng = StdRng::seed_from_u64(0x{seed:08x});\n        println!("seed=0x{seed:08x}");' + body
    tests = tests[:match.end()] + body + tests[end:]
assert len(seeded) == 13
helper_sig = 'fn generate_random_landmarks_ring(n: usize, min_dist: f32, max_dist: f32) -> Vec<Vector2<f32>> {'
assert tests.count(helper_sig) == 1
tests = tests.replace(helper_sig, 'fn generate_random_landmarks_ring(n: usize, min_dist: f32, max_dist: f32, rng: &mut StdRng) -> Vec<Vector2<f32>> {')
tests = tests.replace('rand::random::<f32>()', 'rng.gen::<f32>()')
a = original_ekf.split('#[cfg(test)]\nmod tests {', 1)[1]
for pattern in [r'assert!\([\s\S]*?\);', r'assert_eq!\([\s\S]*?\);', r'const N_TRIALS[^;]*;', r'let n_trials[^;]*;']:
    assert re.findall(pattern, a) == re.findall(pattern, tests), pattern
p.write_text(pre + '#[cfg(test)]\nmod tests {' + tests + '\n#[cfg(test)]\n#[path = "ekf_slam_rng_tests.rs"]\nmod rng_tests;\n')
assert 'rand::random' not in p.read_text()
print('Legacy EKF seeds:', *reversed(seeded), sep='\n')

p = root / 'rust_robotics_algo/src/localization/particle_filter.rs'
assert blob(p.read_bytes()) == 'c64bc160b42b56eeb797ebf650fb9132be24542f'
original_pf = p.read_text()
s = original_pf

def helper(s, name, new_name, params, call, preamble='', replacements=()):
    m = re.search(r'(?m)^(?:pub )?fn ' + name + r'\(', s)
    assert m, name
    brace = s.index('{', m.end())
    end = s.index('\n}', brace) + 2
    header, body = s[m.start():brace + 1], s[brace + 1:end - 1]
    for old, new in replacements:
        assert old in body, (name, old)
        body = body.replace(old, new)
    ret = ('-> ' + header.split('->', 1)[1][:-1].strip()) if '->' in header else ''
    inner = f'fn {new_name}<R: rand::Rng + ?Sized>({params}) {ret} {{\n' + preamble + body + '}'
    return s[:m.start()] + header + '\n    ' + call + '\n}\n\n' + inner + s[end:]

s = helper(s, 'rand', 'rand_with_rng', 'rng: &mut R', 'rand_with_rng(&mut rand::thread_rng())', replacements=[('rand::random::<f32>()', 'rng.gen::<f32>()')])
s = helper(s, 'rand_unifrom', 'uniform_with_rng', 'low: f32, high: f32, rng: &mut R', 'uniform_with_rng(low, high, &mut rand::thread_rng())', replacements=[('    use rand::Rng;\n', ''), ('    let mut rng = rand::thread_rng();\n', '')])
for name, mutable in [('observation', True), ('observation_from_state', False)]:
    ty = '&mut Vector4' if mutable else '&Vector4'
    s = helper(s, name, name + '_with_rng', f'x_true: {ty}, xd: &mut Vector4, u: Vector2, rf_id: &[Vector2], sensing: (f32, f32), rng: &mut R', f'{name}_with_rng(x_true, xd, u, rf_id, (dt, max_range), &mut rand::thread_rng())', preamble='    let (dt, max_range) = sensing;\n', replacements=[('rand()', 'rand_with_rng(rng)')])
s = helper(s, 'pf_localization_with_state', 'pf_localization_with_rng', 'x_est: &mut Vector4, px: &mut PX, pw: &mut PW, z: Vec<Vector3>, control: (Vector2, f32), context: (&mut PFState, &PFNoiseParams), rng: &mut R', 'pf_localization_with_rng(x_est, px, pw, z, (u, dt), (state, noise), &mut rand::thread_rng())', preamble='    let (u, dt) = control;\n    let (state, noise) = context;\n', replacements=[('rand()', 'rand_with_rng(rng)'), ('reset_particles_around_observations(px, pw, &z, x_est);', 'reset_particles_with_rng(px, pw, &z, x_est, rng);'), ('re_sampling(px, pw);', 're_sampling_with_rng(px, pw, rng);')])
s = s.replace('fn reset_particles_around_observations(px: &mut PX, pw: &mut PW, z: &[Vector3], x_est: &Vector4) {', 'fn reset_particles_with_rng<R: rand::Rng + ?Sized>(px: &mut PX, pw: &mut PW, z: &[Vector3], x_est: &Vector4, rng: &mut R) {')
pre, reset = s.split('fn reset_particles_with_rng', 1)
body, post = reset.split('/// Performs low-variance resampling', 1)
s = pre + 'fn reset_particles_with_rng' + body.replace('rand()', 'rand_with_rng(rng)') + '/// Performs low-variance resampling' + post
s = helper(s, 're_sampling', 're_sampling_with_rng', 'px: &mut PX, pw: &mut PW, rng: &mut R', 're_sampling_with_rng(px, pw, &mut rand::thread_rng())', replacements=[('rand_unifrom(0., 1. / NP as f32)', 'uniform_with_rng(0., 1. / NP as f32, rng)')])
s = s[:s.index('#[cfg(test)]\nmod tests {')] + '#[cfg(test)]\n#[path = "particle_filter_seeded_tests.rs"]\nmod seeded_tests;\n'
p.write_text(s)

p = root / 'rust_robotics_sim/src/simulator/common/noise.rs'
s = p.read_text()
original_noise = s
s = s.replace('    2.0 * (rand::random::<f32>() - 0.5)', '''    rand_noise_with_rng(&mut rand::thread_rng())
}

fn rand_noise_with_rng<R: rand::Rng + ?Sized>(rng: &mut R) -> f32 {
    2.0 * (rng.gen::<f32>() - 0.5)''')
s = s.replace('    // Box-Muller transform', '''    gaussian_noise_with_rng(std_dev, &mut rand::thread_rng())
}

fn gaussian_noise_with_rng<R: rand::Rng + ?Sized>(std_dev: f32, rng: &mut R) -> f32 {
    // Box-Muller transform''').replace('rand::random()', 'rng.gen()')
s = s.replace('    use super::*;', '    use super::*;\n    use rand::{rngs::StdRng, SeedableRng};')
s = s.replace('fn test_rand_noise_range() {', 'fn test_rand_noise_range() {\n        let mut rng = StdRng::seed_from_u64(0x5015_0001);').replace('let n = rand_noise();', 'let n = rand_noise_with_rng(&mut rng);')
s = s.replace('fn test_gaussian_noise_distribution() {', 'fn test_gaussian_noise_distribution() {\n        let mut rng = StdRng::seed_from_u64(0x5015_0002);').replace('let n = gaussian_noise(std_dev);', 'let n = gaussian_noise_with_rng(std_dev, &mut rng);')
p.write_text(s)

for name, original in [('rust_robotics_algo/src/slam/ekf_slam.rs', original_ekf), ('rust_robotics_algo/src/localization/particle_filter.rs', original_pf), ('rust_robotics_sim/src/simulator/common/noise.rs', original_noise)]:
    public = lambda text: [re.sub(r'\s+', ' ', sig) for sig in re.findall(r'pub fn [\s\S]*?\{', text)]
    assert public(original) == public((root / name).read_text()), name

p = root / 'scripts/check_numerical_mutations.py'
s = p.read_text()
pos = s.index('\n)\nGROUPS = (')
s = s[:pos] + '''
    Mutation("particle-filter-seed-ignored", "rust_robotics_algo/src/localization/particle_filter.rs",
             "2.0 * (rng.gen::<f32>() - 0.5)", "0.0", "lib",
             "localization::particle_filter::seeded_tests::random_sources_are_seeded_and_replayable",
             ("changing the seed must change the noise stream",)),
    Mutation("ekf-observation-noise-omitted", SLAM + "ekf_slam.rs",
             "rand(rng) * config.observation_noise[(0, 0)].sqrt() * 0.5", "0.0", "lib",
             "slam::ekf_slam::rng_tests::seeded_observations_preserve_noise_formula_and_draw_order",
             ("seeded observation range",)),''' + s[pos:]
s = s.replace('GROUPS = (', '''GROUPS = (
    ("lib", "localization::particle_filter::seeded_tests", 5),
    ("lib", "slam::ekf_slam::rng_tests", 3),''')
p.write_text(s)
print('Prepared bounded RNG plumbing; public signatures, legacy EKF assertions and trial counts unchanged.')
