package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_21_0_Test {

    @Test
    public void testNumberOfRows() {
        // Arrange
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        int expected = 10;
        when(scannerSubscription.numberOfRows()).thenReturn(expected);
        // Act
        int actual = scannerSubscription.numberOfRows();
        // Assert
        assertEquals(expected, actual);
    }
}
