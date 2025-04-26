package net.kencochrane.a4j.file;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { FileUtil_isAgeGood_2_0_Test.class, FileUtil_downloadOneASINFile_0_0_Test.class, FileUtil_deleteFile_1_4_Test.class, FileUtil_renameFile_3_0_Test.class })
public class FileUtil_Suite {
}
