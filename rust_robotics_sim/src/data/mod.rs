pub mod adapter;

pub use adapter::*;

use std::fmt::Debug;
use std::iter::Zip;
use std::ops::RangeInclusive;
use std::slice::Iter;

pub type Time = f32;

/// Maximum number of data entries that [`ColumnData`] and [`TimeTable`] can contain
///
/// For [`ColumnData`], this is equal to the number of data elements
/// For [`TimeTable`], this is equal to the number of rows
///
/// Upon exceeding this limit, [`ColumnData::add`] and [`TimeTable::add`] will
/// remove the first entry to keep size at the limit
pub const LIMIT: usize = 5000;

/// Single column of data
#[derive(Debug, Default, Clone)]
pub struct ColumnData<T> {
    data: Vec<T>,
    name: Option<String>,
}

impl<T> ColumnData<T> {
    pub fn with_name(name: String) -> Self {
        Self {
            data: Vec::new(),
            name: Some(name),
        }
    }
    pub fn from_vec(data: impl Into<Vec<T>>) -> Self {
        Self {
            data: data.into(),
            name: None,
        }
    }

    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn pop_first(&mut self) {
        self.data.remove(0);
    }

    pub fn add(&mut self, element: T) {
        self.data.push(element);
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    pub fn get_between(&self, range: RangeInclusive<usize>) -> &[T] {
        &self.data[range]
    }

    pub fn reset(&mut self) {
        self.data = Vec::new();
    }
}

/// Basic time vector for finding indices to look up within [`TimeSeries`] and [`TimeTable`]
#[derive(Debug, Default)]
struct Timeline {
    /// Cache stores previously found index to avoid unecessary iteration when finding time index
    cache: Option<(Time, usize)>,
    /// Actual vector
    vec: Vec<Time>,
}

impl Into<Timeline> for Vec<Time> {
    fn into(self) -> Timeline {
        Timeline {
            vec: self,
            ..Default::default()
        }
    }
}

impl Timeline {
    /// Tolerance to compare two input time for their equality
    const EPSILON: f32 = 0.0005;

    pub fn new(time_vec: impl Into<Vec<Time>>) -> Self {
        Self {
            vec: time_vec.into(),
            ..Default::default()
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Time> {
        self.vec.iter()
    }

    /// Adds a time element to the end
    pub fn add(&mut self, time: Time) {
        self.vec.push(time);
    }

    pub fn last(&self) -> Time {
        *self.vec.last().unwrap_or(&0.0)
    }

    pub fn pop_first(&mut self) {
        self.vec.remove(0);
        if let Some((_, i)) = &mut self.cache {
            *i -= 1;
        }
    }

    /// Checks if time input has changed from last index search
    /// If time input is sufficiently close, assume same index can be used without calling [`get_index`]
    ///
    /// [`get_index`]: Self::get_index
    fn time_changed(&self, time: Time) -> bool {
        self.cache
            .map_or(true, |(prev, _)| (time - prev).abs() > Self::EPSILON)
    }

    /// Find the index that corresponds to the given time in seconds.
    ///
    /// Returns index of first time that is greater or equal to the specified time.
    fn get_index(&self, time: Time) -> Option<usize> {
        if self.time_changed(time) {
            self.vec.iter().position(|&t| t >= time).map(|index| {
                // self.cache = Some((time, index));
                index
            })
        } else {
            // unwrap here is ok, since time_changed always ensures cache is not None
            Some(self.cache.unwrap().1)
        }
    }

    /// Similar to [`get_index`], but only returns time index that is smaller than the input time.
    /// This is useful when making sure the returned time index never exceeds the given time, as
    /// in [`get_range`]
    ///
    /// [`get_index`]: Self::get_index
    /// [`get_range`]: Self::get_range
    fn get_index_under(&self, time: Time) -> Option<usize> {
        if self.time_changed(time) {
            self.vec
                .iter()
                .position(|&t| t > time)
                .map(|idx| (idx - 1).max(0))
                .map(|index| {
                    // self.cache = Some((time, index));
                    index
                })
        } else {
            // unwrap here is ok, since time_changed always ensures cache is not None
            Some(self.cache.unwrap().1)
        }
    }

    /// Returns range indices that is within the time range specified
    pub fn get_range(&self, start: Time, end: Time) -> Option<RangeInclusive<usize>> {
        if start < end {
            if let Some(start) = self.get_index(start) {
                if let Some(end) = self.get_index_under(end) {
                    return Some(start..=end);
                }
            }
        }
        None
    }

    pub fn get_range_raw(&self, start: Time, end: Time) -> Option<Vec<Time>> {
        self.get_range(start, end)
            .map(|range| self.vec[range].to_vec())
    }

    /// Length of the time vector
    pub fn len(&self) -> usize {
        self.vec.len()
    }
}

#[derive(Debug, Default)]
pub struct TimeTable<T = f32> {
    time: Timeline,
    data: Vec<ColumnData<T>>,
}

impl<T> Into<TimeTable<T>> for TimeSeries<T> {
    fn into(self) -> TimeTable<T> {
        TimeTable {
            time: self.time,
            data: vec![self.data],
        }
    }
}

impl<T: Clone> TimeTable<T> {
    pub fn new(time: Vec<Time>, data: Vec<T>) -> Self {
        TimeSeries::new(time, data).into()
    }

    pub fn from_timeseries(timeseries: TimeSeries<T>) -> Self {
        Self {
            time: timeseries.time,
            data: vec![timeseries.data],
        }
    }
    pub fn names(&self) -> Vec<String> {
        self.data
            .iter()
            .map(|col| col.name().unwrap_or("".to_owned()))
            .collect()
    }
    pub fn ncols(&self) -> usize {
        self.data.len()
    }
    pub fn pop_first(&mut self) {
        self.time.pop_first();
        self.data.iter_mut().for_each(|col| col.pop_first());
    }
    pub fn clear(&mut self) {
        self.time = Timeline::default();
        self.data.iter_mut().for_each(|col| col.reset());
    }

    pub fn get_column(&self, column: usize) -> Option<&ColumnData<T>> {
        self.data.get(column)
    }

    pub fn zipped_iter(&self, column: usize) -> Option<Zip<Iter<'_, f32>, Iter<'_, T>>> {
        self.data
            .get(column)
            .map(|col| self.time.iter().zip(col.iter()))
    }

    pub fn get_at_time(&self, column: usize, time: Time) -> Option<T> {
        if let Some(idx) = self.time.get_index(time) {
            self.data
                .get(column)
                .and_then(|vec| vec.get(idx).clone())
                .map(|el| el.to_owned())
        } else {
            None
        }
    }

    pub fn get_time_range(&self, start: Time, end: Time) -> Option<Vec<Time>> {
        self.time.get_range_raw(start, end)
    }

    pub fn time_last(&self) -> Time {
        self.time.last()
    }

    /// Returns slice of data that is within the time range specified
    pub fn get_range(&self, column: usize, start: Time, end: Time) -> Option<Vec<T>> {
        if let Some(range) = self.time.get_range(start, end) {
            self.data
                .get(column)
                .map(|vec| vec.get_between(range).to_owned())
        } else {
            None
        }
    }
    pub fn init_with_names(names: Vec<&str>) -> Self {
        let data: Vec<ColumnData<T>> = names
            .iter()
            .map(|name| ColumnData::with_name(name.to_string()))
            .collect();
        Self {
            time: Timeline::default(),
            data,
        }
    }
    pub fn nrow(&self) -> usize {
        self.time.len()
    }
    pub fn add(&mut self, time: Time, sample: Vec<T>) {
        self.time.add(time);
        self.data
            .iter_mut()
            .zip(sample.into_iter())
            .for_each(|(vec, el)| vec.add(el));

        if self.nrow() > LIMIT {
            self.pop_first();
        }
    }
}

#[derive(Debug, Default)]
pub struct TimeSeries<T> {
    time: Timeline,
    data: ColumnData<T>,
}

impl<T: Clone> TimeSeries<T> {
    pub fn new(time: impl Into<Vec<Time>>, data: impl Into<Vec<T>>) -> Self {
        let time = Timeline::new(time.into());
        let data = ColumnData::from_vec(data);

        if time.len() != data.len() {
            panic!("Size of time and data are different!");
        }
        Self { time, data }
    }

    pub fn iter(&self) -> Zip<Iter<'_, f32>, Iter<'_, T>> {
        self.time.iter().zip(self.data.iter())
    }

    pub fn with_name(self, name: String) -> Self {
        let Self { time, mut data, .. } = self;
        data.name = Some(name);
        Self { time, data }
    }

    pub fn empty() -> Self {
        Self {
            time: Timeline::default(),
            data: ColumnData {
                data: Vec::new(),
                name: None,
            },
        }
    }

    pub fn add(&mut self, time: Time, element: T) {
        self.time.add(time);
        self.data.add(element);
    }

    pub fn time_last(&self) -> Time {
        self.time.last()
    }

    /// Get data element for a given time
    pub fn get_at_time(&self, time: Time) -> Option<T> {
        self.time
            .get_index(time)
            .and_then(|idx| self.data.get(idx))
            .map(|val| val.to_owned())
    }

    /// Returns slice of data that is within the time range specified
    pub fn get_range(&self, start: Time, end: Time) -> Option<&[T]> {
        self.time
            .get_range(start, end)
            .map(|range| self.data.get_between(range))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_f32() -> TimeSeries<f32> {
        let n = 5;
        let dt = 1.0;
        let t: Vec<f32> = (0..n).map(|n| n as f32 * dt).collect();
        let data: Vec<f32> = t.iter().map(|&t| t * 3.0).collect();

        TimeSeries::new(t, data)
    }

    #[test]
    fn add_timeseries() {
        let mut ts = TimeSeries::<f32>::empty();
        ts.add(0.5, 5.0);
        ts.add(0.8, 15.0);
        dbg!(&ts);
    }

    #[test]
    fn check_index() {
        // dbg!(&ts);
        let ts = dummy_f32();

        assert_eq!(2, ts.time.get_index(2.0).unwrap()); // finding exactly matching time
        assert_eq!(2, ts.time.get_index(2.0).unwrap()); // running again should give same result
        assert_eq!(2, ts.time.get_index(1.5).unwrap()); // finding next closest time stamp
    }

    #[test]
    fn check_range() {
        let ts = dummy_f32();
        assert_eq!(1, ts.time.get_index(1.0).unwrap());
        assert_eq!(3, ts.time.get_index(2.1).unwrap());
        assert_eq!(3, ts.time.get_index(2.9).unwrap());
        assert_eq!(3, ts.time.get_index(3.0).unwrap());
        assert_eq!(&[3.0, 6.0], ts.get_range(1.0, 2.9).unwrap());
        assert_eq!(&[3.0, 6.0, 9.0], ts.get_range(1.0, 3.0).unwrap());
    }

    #[test]
    fn series_to_table() {
        let _ts = dummy_f32();
        // let _table: TimeTable<f32> = ts.into();
    }
}
