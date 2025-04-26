package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_numberOfRows_0_0_Test {

    @Mock
    private ScannerSubscription focal;

    @InjectMocks
    private ScannerSubscription testInstance;

    @Test
    public void testNumberOfRows() {
        // Arrange
        // Given
        // When
        int expectedNumberOfRows = 5;
        when(focal.numberOfRows()).thenReturn(expectedNumberOfRows);
        // Act
        int actualNumberOfRows = testInstance.numberOfRows();
        // Assert
        assertEquals(expectedNumberOfRows, actualNumberOfRows);
    }
}
