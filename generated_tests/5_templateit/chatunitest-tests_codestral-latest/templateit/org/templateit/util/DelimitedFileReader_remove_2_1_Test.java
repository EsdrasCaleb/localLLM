package org.templateit.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.NoSuchElementException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.FileReader;
import java.util.Iterator;

@ExtendWith(MockitoExtension.class)
public class DelimitedFileReader_remove_2_1_Test {

    @InjectMocks
    private DelimitedFileReader delimitedFileReader;

    @Mock
    private BufferedReader reader;

    @BeforeEach
    public void setUp() throws FileNotFoundException {
        delimitedFileReader = new DelimitedFileReader(new File("test.txt"), ",");
    }

    @Test
    public void testRemove_2_1() throws IOException {
        // Mock the behavior of the BufferedReader
        when(reader.readLine()).thenReturn("1,2,3").thenReturn(null);
        // Set the reader field using reflection
        try {
            java.lang.reflect.Field field = DelimitedFileReader.class.getDeclaredField("reader");
            field.setAccessible(true);
            field.set(delimitedFileReader, reader);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        // Call the remove method
        assertThrows(UnsupportedOperationException.class, () -> delimitedFileReader.remove());
    }

    @Test
    public void testRemove() {
        assertThrows(UnsupportedOperationException.class, () -> {
            delimitedFileReader.remove();
        });
    }
}
