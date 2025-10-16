
import os
from ij import IJ, ImagePlus, WindowManager
from ij.gui import GenericDialog, DialogListener
from ij.plugin import Duplicator
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from loci.formats import ImageReader
import time
import csv

#@ File (label="Select .lif file", style="file") input_file
        
def grab_stacks(input_file):

    if input_file.getAbsolutePath().lower().endswith(".lif"):
        
        input_file = input_file.getAbsolutePath()

        # Read the .lif file and get the number of series
        reader = ImageReader()
        try:
            reader.setId(input_file)
            series_number = reader.getSeriesCount()
        finally:
            reader.close()

        IJ.log("Found " + str(series_number) + " series in " + os.path.basename(input_file))

        base, ext = os.path.splitext(os.path.basename(input_file))
        main_name = base + "_stats.csv"
        main_path = os.path.join(os.path.dirname(input_file), main_name)
        os.remove(main_path) if os.path.exists(main_path) else None


        for series in range(series_number):

            series_options = ImporterOptions()
            series_options.setId(input_file)
            for i in range(series_number):
                series_options.setSeriesOn(i, False)
            series_options.setSeriesOn(series, True)
            series_options.setVirtual(True)
        
            imps = BF.openImagePlus(series_options)

            if imps is None or len(imps) == 0:
                IJ.log("Could not open series " + str(series) + " from " + os.path.basename(input_file))
                continue

            # Show the first image in the series
            imp = imps[0]
            imp.show()

            IJ.run("Brightness/Contrast...")

            gd_bc = GenericDialog("Adjust Contrast and Crop " + imp.getTitle())
            gd_bc.addMessage("Use the 'Maximum Intensity' slider in the 'B&C' window to adjust the contrast." \
                             "\nClick OK when finished.")

            class ContrastDialogListener(DialogListener):
                def dialogItemChanged(self, gd_bc, e):
                    if gd_bc.wasOKed() or gd_bc.wasCanceled():
                        b_c = WindowManager.getWindow("B&C")
                        if b_c:
                            b_c.close()
                        return True

                def dialogCancelled(self, gd_bc):
                    return True
                
            gd_bc.addDialogListener(ContrastDialogListener())
            gd_bc.setModal(False)
            gd_bc.showDialog()                
            
            if gd_bc.wasCanceled():
                IJ.log("Processing canceled by user.")
                imp.close()
                return

            slices = imp.getNSlices()
            channels = imp.getNChannels()
            frames = imp.getNFrames()

            gd_slice = GenericDialog("Process Series " + str(series + 1) + ": " + imp.getTitle())
            gd_slice.addSlider("Select slice", 1, slices, imp.getSlice())
            gd_slice.addSlider("Channel:", 1, channels, imp.getChannel())

            class SliderListener(DialogListener):
                def dialogItemChanged(self, gd_slice, e):
                    if gd_slice.wasOKed() or gd_slice.wasCanceled():
                        return True
                    
                    gd_slice.resetCounters()
                    slice = int(gd_slice.getNextNumber())
                    channel = int(gd_slice.getNextNumber())
                    imp.setPosition(channel, slice, imp.getFrame())
                    imp.updateAndDraw()
                    return True

                def dialogCancelled(self, gd_slice):
                    return True

            gd_slice.addDialogListener(SliderListener())
            gd_slice.setModal(False)
            gd_slice.showDialog()
            while gd_bc.isShowing() or gd_slice.isShowing(): time.sleep(0.1)
            if gd_slice.wasCanceled():
                IJ.log("Processing canceled by user.")
                imp.close()
                b_c = WindowManager.getWindow("B&C")
                if b_c:
                    b_c.close()
                continue

            gd_finish = GenericDialog("Crop Image.")
            gd_finish.addMessage("\nUse your cursor to drag a box over the region you want to crop." \
                                    "\nClick OK when finished to save unprocessed crop.")
            gd_finish.setModal(False)
            gd_finish.showDialog()
            while gd_finish.isShowing(): time.sleep(0.1)

            gd_slice.resetCounters()
            slice = int(gd_slice.getNextNumber())
            channel = int(gd_slice.getNextNumber())
            slice_imp = Duplicator().run(imp, channel, channel, slice, slice, 1, frames)

            if slice_imp is not None and isinstance(slice_imp, ImagePlus):
                imp.close()
                slice_imp.setTitle("Series_" + str(series + 1) + "_Slice_" + str(slice))
                slice_imp.show()

                WindowManager.setCurrentWindow(slice_imp.getWindow())
                IJ.run(slice_imp, "Apply LUT", "")


                IJ.run(slice_imp, "Subtract Background...", "rolling=50 light=none stack=none stack_hist=none")
                IJ.run(slice_imp, "Apply LUT", "")

                # Save unprocessed cropped image
                base, ext = os.path.splitext(os.path.basename(input_file))
                output_name = base + "_series_" + str(series + 1) + "_slice_" + str(slice) + ".tif"
                output_path = os.path.join(os.path.dirname(input_file), output_name)
                IJ.saveAs(slice_imp, "Tiff", output_path)
                IJ.log("Saved slice " + str(slice) + " of series " + str(series + 1) + " to " + output_path)

                IJ.run("Window/Level...")

                gd_wl = GenericDialog("Adjust Window and Level " + slice_imp.getTitle())
                gd_wl.addMessage("Use the sliders in the 'W&L' window to adjust the contrast." \
                                "\nClick OK when finished to save binary image.")

                class WindowDialogListener(DialogListener):
                    def dialogItemChanged(self, gd_wl, e):
                        if gd_wl.wasOKed() or gd_wl.wasCanceled():
                            w_l = WindowManager.getWindow("W&L")
                            if w_l:
                                w_l.close()
                            return True

                    def dialogCancelled(self, gd_wl):
                        return True

                gd_wl.addDialogListener(WindowDialogListener())
                gd_wl.setModal(False)
                gd_wl.showDialog()
                while gd_wl.isShowing(): time.sleep(0.1)

                if gd_wl.wasCanceled():
                    IJ.log("Processing canceled by user.")
                    slice_imp.close()
                    w_l = WindowManager.getWindow("W&L")
                    if w_l:
                        w_l.close()
                    continue
                
                WindowManager.setCurrentWindow(slice_imp.getWindow())
                IJ.run(slice_imp, "Apply LUT", "")

                # Save binary image
                base, ext = os.path.splitext(os.path.basename(input_file))
                output_name = base + "_series_" + str(series + 1) + "_slice_" + str(slice) + "_binary.tif"
                output_path = os.path.join(os.path.dirname(input_file), output_name)
                IJ.saveAs(slice_imp, "Tiff", output_path)
                IJ.log("Saved processed image to " + output_path)

                IJ.run(slice_imp, "32-bit", "")
                IJ.run(slice_imp, "ROF Denoise", "theta=25.0 iterations=1")
                IJ.run(slice_imp, "8-bit", "")

                IJ.run(slice_imp, "Apply LUT", "")
                
                IJ.run(slice_imp, "Skeletonize (2D/3D)", "")
                IJ.run(slice_imp, "Analyze Skeleton (2D/3D)", "prune=none calculate")

                # Save skeleton results 
                rt = WindowManager.getWindow("Results")
                if rt:
                    base, ext = os.path.splitext(os.path.basename(input_file))
                    results_name = base + "_series_" + str(series + 1) + "_slice_" + str(slice) + "_skeleton_results.csv"
                    results_path = os.path.join(os.path.dirname(input_file), results_name)
                    IJ.selectWindow("Results")
                    IJ.saveAs("Results", results_path)
                    IJ.log("Saved skeleton results to " + main_path)
                    rt.close()
                    with open(main_path, 'a') as f_main, open(results_path, 'r') as f_results:
                        f_main.write("\n\nResults for " + base + "_series_" + str(series + 1) + "_slice_" + str(slice) + ":\n")
                        for line in f_results:
                            f_main.write(line)
                    analyze_results(results_path, main_path)

                gd_finish = GenericDialog("Skeleton processing.")
                gd_finish.addMessage("Click OK when done examining skeleton.")
                gd_finish.setModal(False)
                gd_finish.showDialog()
                while gd_finish.isShowing(): time.sleep(0.1)

                tskel = WindowManager.getImage("Tagged skeleton")
                if tskel:
                    base, ext = os.path.splitext(os.path.basename(input_file))
                    output_name = base + "_series_" + str(series + 1) + "_slice_" + str(slice) + "_tagged.tif"
                    output_path = os.path.join(os.path.dirname(input_file), output_name)
                    IJ.selectWindow("Tagged skeleton")
                    IJ.saveAs(tskel, "Tiff", output_path)
                    IJ.log("Saved tagged skeleton to " + output_path)
                    tskel.close()
                lspaths = WindowManager.getImage("Longest shortest paths")
                if lspaths:
                    lspaths.close()

                base, ext = os.path.splitext(os.path.basename(input_file))
                output_name = base + "_series_" + str(series + 1) + "_slice_" + str(slice) + "_processed.tif"
                output_path = os.path.join(os.path.dirname(input_file), output_name)
                IJ.saveAs(slice_imp, "Tiff", output_path)
                IJ.log("Saved processed image to " + output_path)
                slice_imp.close()
            else:
                IJ.log("Slice extraction failed.")

            imp.close()

        IJ.log("Processing complete.")
        return


def analyze_results(results_path, main_path):

    if not results_path:
        IJ.log("No results to analyze.")
        return

    column_1_sum = 0
    column_2_sum = 0
    with open(results_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [header]
        for row in reader:
            try:
                column_1_sum += float(row[0])
                column_2_sum += float(row[1])
            except ValueError:
                IJ.log("Non-numeric data found in results.")

    ratio = column_2_sum / column_1_sum if column_1_sum != 0 else 0

    # Append ratio to the end of the CSV without altering previous data
    with open(main_path, 'a') as f:
        writer = csv.writer(f)
        #writer.writerow([])
        #writer.writerow([results_path])
        writer.writerow(["Integrity (J/B)", ratio])
    #IJ.log("Analysis complete. Summary saved to " + summary_path)
    # delete original results file
    os.remove(results_path)

grab_stacks(input_file)
log = WindowManager.getWindow("Log")
if log:
    log.close()