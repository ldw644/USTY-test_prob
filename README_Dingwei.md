# Usage
- Fewer than 20 new lines have been added; most of the functions are generally maintained.
- Now we need to call `MergedCurve(df_curve_python, df_curve_fortran)` to generate a `MergedCurve` object. We use `MergedCurve.plot_fwdrate_wrapper` to generate images.
- Additional functions should be placed within the `MergedCurve` class.
- Generated images are located in the `/output/merged` directory.

# Known issue
- Some data is missing in the original Fortran `.pkl` file (e.g., yvols). Therefore, some parameters are not calculated.


# Some further modification possibilities...
- I didn't make significant modifications for this test. I only provided the necessary changes to complete it. However, I personally believe that some improvements will be essential in the near future -- regardless of whether I get this position.
- Integrate a configuration file (e.g., JSON or YAML) to manage input parameters without modifying the code (current Option-related inputs section).
- Greater extensibility is needed. If we plan to publish the code in the future, we may need to allow users to manually input data for various parameters (e.g., date, rate, etc.). Now would be a good time to separate inputs by columns, rather than inputting a large dataframe.
- I also recommend breaking down long functions into smaller, more focused ones, each handling a single responsibility. This will improve the readability, maintainability, and testability of the code by making each function easier to understand and debug. `pd.pipe` is a good option for this.