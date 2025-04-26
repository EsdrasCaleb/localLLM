// Test method
package org.templateit.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.NoSuchElementException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class DelimitedFileReader_hasNext_0_0_Test {

    @Test
    public void testHasNext() throws Exception {
        // Arrange
        DelimitedFileReader reader = mock(DelimitedFileReader.class);
        when(reader.hasNext()).thenReturn(true).thenReturn(false);
        // Act
        boolean hasNext = reader.hasNext();
        // Assert
        assertTrue(hasNext);
        assertFalse(reader.hasNext());
    }
}
