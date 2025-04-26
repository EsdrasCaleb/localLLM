package org.templateit.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.Reader;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class DelimitedFileReader_hasNext_0_0_Test {

    @Mock
    private DelimitedFileReader delimitedFileReader;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        File file = new File("test.txt");
        try {
            delimitedFileReader = new DelimitedFileReader(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void hasNext_returnsTrue_whenNextLineIsNotNull() {
        when(delimitedFileReader.hasNext()).thenReturn(true);
        assertTrue(delimitedFileReader.hasNext());
    }

    @Test
    public void hasNext_returnsFalse_whenNextLineIsNull() {
        when(delimitedFileReader.hasNext()).thenReturn(null);
        assertFalse(delimitedFileReader.hasNext());
    }
}
