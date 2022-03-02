# Inputs for all the case studies

input_fix =  {
              "filtered_filename": "",
              "filter_dx":         8,
              "filter_npass":      1,
              "input_dir":   "/work/larissa.reames",
              "output_dir":  "/work/wicker/CAM_case_studies",
              "fprefix":     "econus",
              "klevels":     None,
              "writeout":    True,
              "cases": {
                       #"2019071918": ["hrrr", "ctrl", "nord3", "nam"],
                       #"2020081006": ["hrrr", "ctrl", "nord3", "nam"],
                       #"2020070700": ["hrrr", "ctrl", "nord3", "nam"],
                       #"2020030212": ["hrrr", "ctrl", "nord3", "nam"],
                       #"2020050300": ["hrrr", "ctrl", "nord3", "nam"],
                       #"2021090100": ["hrrr", "ctrl", "nord3", "nam"],
                        "2021052612": ["hrrr", "ctrl"]
                        },
               "zoom": {
                        "2019071918": [44.0, 49.0,  -92.0, -87.0, 5],
                        "2020081006": [39.0, 44.0,  -92.0, -86.0, 5],
                        "2020030212": [34.0, 39.5,  -92.0, -85.0, 2],
                        "2021090100": [35.5, 43.0,  -80.0, -73.0, 5],
                        "2020050300": [35.0, 40.0,  -92.0, -85.0, 5],
                        "2020070700": [42.0, 46.0, -101.0, -96.0, 5],
                        "2021052612": [33.5, 43.0, -102.5, -97.0, 5]
                        }
              }

input_all =  {
              "filtered_filename": "",
              "filter_dx":         12,
              "filter_npass":      1,
              "input_dir":   "/work/larissa.reames",
              "output_dir":  "/work/wicker/CAM_case_studies",
              "fprefix":     "econus",
              "klevels":     None,
              "writeout":    True,
              "cases": {
                        #"2019071918": ["hrrr", "ctrl", "nord3", "nam"],
                        "2020081006": ["hrrr", "ctrl", "nord3", "nam"],
                        "2020070700": ["hrrr", "ctrl", "nord3", "nam"],
                        "2020030212": ["hrrr", "ctrl", "nord3", "nam"],
                        "2020050300": ["hrrr", "ctrl", "nord3", "nam"],
                        "2021090100": ["hrrr", "ctrl", "nord3", "nam"],
                        "2021052612": ["hrrr", "ctrl", "nord3", "nam"],
                        },
               "zoom": {
                        "2019071918": [44.0, 49.0,  -92.0, -87.0, 5],
                        "2020081006": [39.0, 44.0,  -92.0, -86.0, 5],
                        "2020030212": [34.0, 39.5,  -92.0, -85.0, 2],
                        "2021090100": [35.5, 43.0,  -80.0, -73.0, 5],
                        "2020050300": [35.0, 40.0,  -92.0, -85.0, 5],
                        "2020070700": [42.0, 46.0, -101.0, -96.0, 5],
                        "2021052612": [33.5, 43.0, -102.5, -97.0, 5]
                        }
              }

# Input for a single model set of runs

input_nam =  {
              "filtered_filename": "",
              "filter_dx":          12,
              "filter_npass":       1,
              "input_dir":   "/work/larissa.reames",
              "output_dir":  "/work/wicker/CAM_case_studies",
              "fprefix":     "econus",
              "klevels":     None,
              "writeout":    False,
              "cases": {
                        "2020081006": ["nam"],
                        "2020070700": ["nam"],
                        "2020030212": ["nam"],
                        "2020050300": ["nam"],
                        "2021090100": ["nam"],
                        "2021052612": ["nam"],
                       },
               "zoom": {
                        "2021052612": [33.5, 43.0, -103.0, -97.0, 5],
                        "2020081006": [39.0, 44.0,  -92.0, -86.0, 5],
                        "2020030212": [34.0, 39.5,  -92.0, -85.0, 2],
                        "2021090100": [35.5, 43.0,  -80.0, -73.0, 5],
                        "2020050300": [35.0, 40.0,  -92.0, -85.0, 5],
                        "2020070700": [42.0, 46.0, -101.0, -96.0, 5],
                       }
              }


# Inputtest

input_test =  {
              "filtered_filename": "",
              "filter_dx":          8,
              "filter_npass":       1,
              "input_dir":   "/work/larissa.reames",
              "output_dir":  "/work/wicker/CAM_case_studies",
              "fprefix":     "econus",
              "klevels":     [25],      # process a single level (fast)
              "writeout":    False,     # this is a test, dont write a new file.
              "cases": {
                        "2021052612": ["hrrr"],
                       },
               "zoom": {
                        "2021052612": [33.5, 43.0, -103.0, -97.0, 5]
                       }
              }

