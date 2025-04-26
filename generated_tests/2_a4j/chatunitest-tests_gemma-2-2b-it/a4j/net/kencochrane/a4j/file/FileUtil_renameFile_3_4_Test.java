package net.kencochrane.a4j.file;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

public class FileUtil_renameFile_3_4_Test {

    @Test
    void renameFile() {
        FileUtil fileUtil = new FileUtil();
        fileUtil.cacheDir = "test";
        fileUtil.oldestAge = 100;
        String oldFileName = "test.txt";
        String newFileName = "new.txt";
        fileUtil.renameFile(oldFileName, newFileName);
    }
}
