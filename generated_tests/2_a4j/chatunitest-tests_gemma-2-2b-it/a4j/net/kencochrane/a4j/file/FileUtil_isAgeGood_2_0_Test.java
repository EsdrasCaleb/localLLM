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

public class FileUtil_isAgeGood_2_0_Test {

    @Test
    void isAgeGood_shouldReturnTrueIfFileIsGood() {
        FileUtil fileUtil = mock(FileUtil.class);
        when(fileUtil.isAgeGood(any(File.class))).thenReturn(true);
        assertTrue(fileUtil.isAgeGood(new File("testFile.txt")));
    }

    @Test
    void isAgeGood_shouldReturnFalseIfFileIsBad() {
        FileUtil fileUtil = mock(FileUtil.class);
        when(fileUtil.isAgeGood(any(File.class))).thenReturn(false);
        assertFalse(fileUtil.isAgeGood(new File("testFile.txt")));
    }

    @Test
    void isAgeGood_shouldReturnFalseIfFileIsNull() {
        FileUtil fileUtil = mock(FileUtil.class);
        when(fileUtil.isAgeGood(any(File.class))).thenReturn(false);
        assertFalse(fileUtil.isAgeGood(null));
    }
}
