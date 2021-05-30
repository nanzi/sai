from tfprocess import TFProcess
tfprocess = TFProcess(12, 256, 0.05, 1000, 1000, 1000, 5, 1, 1, 0, 0, 1, 2, 0)
tfprocess.init(128, logbase="test00", macrobatch=4)
tfprocess.restore("../g26d-76897f32-4920h")
tfprocess.save_leelaz_weights("prova")
