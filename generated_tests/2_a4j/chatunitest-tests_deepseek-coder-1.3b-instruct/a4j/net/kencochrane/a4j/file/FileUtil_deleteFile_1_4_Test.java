package net.kencochrane.a4j.file;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

@ExtendWith(MockitoExtension.class)
class FileUtil_deleteFile_1_4_Test {

    @Mock
    FileUtil fileUtil;

    @BeforeEach
    void setUp() {
        Field[] fields = FileUtil.class.getDeclaredFields();
        for (Field field : fields) {
            field.setAccessible(true);
            if (field.getName().equals("cacheDir")) {
                try {
                    field.set(fileUtil, "/tmp");
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
            }
            if (field.getName().equals("oldestAge")) {
                try {
                    field.set(fileUtil, 10L);
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    @Test
    void deleteFile() throws NoSuchMethodException {
        // Arrange
        String fileName = "test.txt";
        // Act
        doNothing().when(fileUtil).deleteFile(fileName);
        // Assert
        assertDoesNotThrow(() -> fileUtil.deleteFile(fileName));
    }
}
