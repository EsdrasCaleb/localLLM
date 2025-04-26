package net.kencochrane.a4j.file;

import java.io.File;
import java.util.Date;
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
import java.util.Properties;
import java.util.Random;

public class FileUtil_isAgeGood_2_0_Test {

    @Test
    public void testIsAgeGood() {
        FileUtil fileUtil = new FileUtil();
        File mockFile = Mockito.mock(File.class);
        Date mockDate = Mockito.mock(Date.class);
        when(mockFile.lastModified()).thenReturn(new Date().getTime());
        when(mockDate.getTime()).thenReturn(new Date().getTime());
        when(mockFile.length()).thenReturn(500L);
        when(mockDate.getTime()).thenReturn(new Date().getTime() - 1000L);
        Assertions.assertTrue(fileUtil.isAgeGood(mockFile));
        Assertions.assertFalse(fileUtil.isAgeGood(mockFile));
    }
}
