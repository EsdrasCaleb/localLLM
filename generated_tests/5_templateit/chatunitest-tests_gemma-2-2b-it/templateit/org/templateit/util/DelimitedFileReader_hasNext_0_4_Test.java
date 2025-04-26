package org.templateit.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class DelimitedFileReader_hasNext_0_4_Test {

    @Test
    void hasNext() {
        DelimitedFileReader reader = Mockito.mock(DelimitedFileReader.class);
        Mockito.when(reader.hasNext()).thenReturn(true);
        assertTrue(reader.hasNext());
        Mockito.verify(reader).hasNext();
    }
}
