package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_0_0_Test {

    @Test
    public void testNumberOfRows() {
        // Arrange
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        when(scannerSubscription.numberOfRows()).thenReturn(10);
        // Act
        int numberOfRows = scannerSubscription.numberOfRows();
        // Assert
        assertEquals(10, numberOfRows);
    }
}
