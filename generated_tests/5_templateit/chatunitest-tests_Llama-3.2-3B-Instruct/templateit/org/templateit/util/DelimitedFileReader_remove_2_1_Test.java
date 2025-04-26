package org.templateit.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.NoSuchElementException;

@ExtendWith(MockitoExtension.class)
public class DelimitedFileReader_remove_2_1_Test {

    @InjectMocks
    private DelimitedFileReader reader;

    @Test
    public void testRemove_throwsUnsupportedOperationException() {
        // Arrange
        // Given
        // When
        // Then
        assertThrows(UnsupportedOperationException.class, () -> reader.remove());
    }
}
