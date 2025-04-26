package net.kencochrane.a4j.file;

import java.io.File;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

public class FileUtil_renameFile_3_0_Test {

    @Test
    public void testRenameFile() {
        FileUtil fileUtil = new FileUtil();
        File oldFile = new File("test.txt");
        File newFile = new File("test1.txt");
        assertTrue(oldFile.renameTo(newFile));
    }
}
